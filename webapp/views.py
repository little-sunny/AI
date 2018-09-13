from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from django.template.context import RequestContext
import os

# Create your views here.


def remve_tempfile(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            os.rmdir(path_file)


def upload(request):
    if request.method == 'GET':
        return render(request, 'upload.html')
    elif request.method == 'POST':
        content = request.FILES.get("upload", None)
        if not content:
            return HttpResponse("NO Content")

        path = 'D:\source\python\AI\Demo\data'
        remve_tempfile(path)
        position = os.path.join(path, content.name)

        storage = open(position, 'wb+')  #打开存储文件

        for chunk in content.chunks():
            storage.write(chunk)
        storage.close()
        result = "上传成功"
        return render_to_response('train.html', RequestContext(request, {'upload_result': result}))
    else:
        return HttpResponse("上传失败")


def do_noramlmap(request):
    return HttpResponse("This is normal maping.")