#The result of third assignment<br>
<pre>
我从一个网站拷贝了测试数据集，以下是其链接：
https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMjc4OF9WZWhpY2xlIERhdGFzZXQgZm9yIFlPTE8vdmVoaWNsZS1kYXRhc2V0LWZvci15b2xvLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogInRtZEFaaXVzQXZPQkNySVc1L1dXZjVicVY0aS9iUVNnOWJaZlFQMlpsWU09In0=?response-content-disposition=attachment%3B%20filename%3D%22vehicle-dataset-for-yolo-DatasetNinja.tar%22
我训练了6个标签，分别是"bus","car","motorbike","threewheel","truck","van"，测试结果如下：
 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
                   all        311        375      0.199      0.354      0.232     0.0763  
                   bus         57         66      0.206      0.758      0.545      0.174  
                   car         42         46      0.125      0.109      0.105      0.029  
             motorbike         74         85     0.0701      0.565      0.154     0.0561  
            threewheel         68         74      0.285       0.23      0.201     0.0645  
                 truck         50         64      0.211      0.391       0.23     0.0819  
                   van         40         40      0.298      0.075      0.158     0.0529  
由于我只进行了两轮训练，同时数据集较小，因此结果不佳
<pre>
