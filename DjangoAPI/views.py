from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from .models import Customer
from .serializer import CustomerSerializers
import paramiko
from django.shortcuts import render
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import io
import __main__


def execute_capturesh():
    script_path = '/home/alext/capture.sh'

    # Подключаемся к Raspberry Pi по SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('raspberrypi.local', username='alext', key_filename='DjangoAPI/priv.ppk', password='kiko91515')

    # Запускаем скрипт на Raspberry Pi
    stdin, stdout, stderr = client.exec_command(script_path)

    # Ожидаем завершения выполнения скрипта
    output = stdout.read().decode('utf-8')

    # Закрываем соединение
    client.close()

    return output


def execute_runfactoryio():
    # Получаем данные из POST-запроса
    script_path = '/home/alext/runfactotyio.sh'

    # Подключаемся к Raspberry Pi по SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('raspberrypi.local', username='alext', key_filename='DjangoAPI/priv.ppk', password='kiko91515')

    # Запускаем скрипт на Raspberry Pi
    stdin, stdout, stderr = client.exec_command(script_path)

    # Ожидаем завершения выполнения скрипта
    output = stdout.read().decode('utf-8')

    # Закрываем соединение
    client.close()

    return output


SCOPES = ['https://www.googleapis.com/auth/drive',
          'https://www.googleapis.com/auth/drive.appdata',
          'https://www.googleapis.com/auth/drive.file',
          'https://www.googleapis.com/auth/drive.metadata',
          'https://www.googleapis.com/auth/drive.metadata.readonly',
          'https://www.googleapis.com/auth/drive.photos.readonly',
          'https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'DjangoAPI/phandomat1-1717017098189-63059c04887a.json'
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
setattr(__main__, "JSON_KEY_FILE", SERVICE_ACCOUNT_FILE)


def initialize_drive():
    """Initializes an drive service object.

  Returns:
    An authorized drive service object.
  """
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    # Build the service object.
    service = build('drive', 'v3', credentials=credentials)

    return service


def download_report(drive_service):
    results = drive_service.files().list(
        pageSize=10,
        fields="nextPageToken, files(id, name, mimeType, parents, createdTime)",
        q="'1-XFKLc6ofm1nSvS1oa-vOfCkbPu2d3qs' in parents").execute()
    file_id = results.get('files')[0].get("id")
    request = drive_service.files().get_media(fileId=file_id)
    filename = 'media/images/' + file_id + '.jpg'
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    my_file = open(filename, 'wb')
    my_file.write(fh.getvalue())
    my_file.close()

    return filename


class CustomerView(viewsets.ModelViewSet):
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializers


transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def status(image, class_of_garbage):
    try:
        def accuracy(outputs, labels):
            _, preds = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(preds == labels).item() / len(preds))

        class ImageClassificationBase(nn.Module):
            def training_step(self, batch):
                images, labels = batch
                out = self(images)  # Generate predictions
                loss = F.cross_entropy(out, labels)  # Calculate loss
                return loss

            def validation_step(self, batch):
                images, labels = batch
                out = self(images)  # Generate predictions
                loss = F.cross_entropy(out, labels)  # Calculate loss
                acc = accuracy(out, labels)  # Calculate accuracy
                return {'val_loss': loss.detach(), 'val_acc': acc}

            def validation_epoch_end(self, outputs):
                batch_losses = [x['val_loss'] for x in outputs]
                epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
                batch_accs = [x['val_acc'] for x in outputs]
                epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
                return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

            def epoch_end(self, epoch, result):
                print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                    epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))

        class ResNet(ImageClassificationBase):
            def __init__(self):
                super().__init__()
                # Use a pretrained model
                self.network = models.resnet50(pretrained=True)
                # Replace last layer
                num_ftrs = self.network.fc.in_features
                self.network.fc = nn.Linear(num_ftrs, 6)

            def forward(self, xb):
                return torch.sigmoid(self.network(xb))

        model = ResNet()

        class DeviceDataLoader():
            """Wrap a dataloader to move data to a device"""

            def __init__(self, dl, device):
                self.dl = dl
                self.device = device

            def __iter__(self):
                """Yield a batch of data after moving it to device"""
                for b in self.dl:
                    yield to_device(b, self.device)

            def __len__(self):
                """Number of batches"""
                return len(self.dl)

        def to_device(data, device):
            """Move tensor(s) to chosen device"""
            if isinstance(data, (list, tuple)):
                return [to_device(x, device) for x in data]
            return data.to(device, non_blocking=True)

        def get_default_device():
            """Pick GPU if available, else CPU"""
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')

        device = get_default_device()
        setattr(__main__, "ResNet", ResNet)
        model = to_device(ResNet(), device)
        model = torch.load('DjangoAPI/garbage_network_entire.h5', map_location=torch.device(device))
        model.eval()
        image = Image.open(image)
        example_image = transformations(image)
        # Convert to a batch of 1
        xb = to_device(example_image.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        prob, preds = torch.max(yb, dim=1)
        # Retrieve the class label
        dataset_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        # plt.imshow(example_image.permute(1, 2, 0))

        if class_of_garbage == '1':
            return "Да" if dataset_classes[preds[0].item()] == 'plastic' else "No"
        elif class_of_garbage == '2':
            return "Да" if dataset_classes[preds[0].item()] == 'glass' else "No"
        elif class_of_garbage == '3':
            return "Да" if dataset_classes[preds[0].item()] == 'metal' else "No"

    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def FormView(request):
    if request.method == "POST":
        execute_capturesh()
        content = download_report(initialize_drive())
        Image_for_sort = content
        if 'plastic' in request.POST:
            Class_of_garbage = 1  # do subscribe
        elif 'glass' in request.POST:
            Class_of_garbage = 2
            # do unsubscribe
        elif 'metal' in request.POST:
            Class_of_garbage = 3
        obj = Customer.objects.create(
            class_of_garbage=Class_of_garbage,
            image=Image_for_sort
        )
        obj.save()
        result = status(Image_for_sort, Class_of_garbage)
        if result == 'Да':
            execute_runfactoryio()
            result = 'Совпадение с ответом нейронной сети. Поздравляю!'
        else:
            result = "Класс не совпадает. Пожалуйста, заберите объект! Спасибо, что заботитесь о природе!"
        return render(request, 'status.html', {"data": result})

    return render(request, 'form.html')
