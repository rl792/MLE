[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/5gWs3A8E)
# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py project/run_fast_tensor.py project/parallel_check.py tests/test_tensor_general.py

Log for run_sentiment.py (epoch=25, learning rate=0.01):
```
$ py run_sentiment.py 
Reusing dataset glue (C:\Users\Rosyy\.cache\huggingface\datasets\glue\sst2\1.0.0\dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 25.31it/s]
missing pre-trained embedding for 55 unknown words
Epoch 1, loss 31.505909272359826, train accuracy: 51.11%
Validation accuracy: 51.00%
Best Valid accuracy: 51.00%
Epoch 2, loss 31.279409316428474, train accuracy: 49.11%
Validation accuracy: 48.00%
Best Valid accuracy: 51.00%
Epoch 3, loss 31.008566839608278, train accuracy: 56.89%
Validation accuracy: 55.00%
Best Valid accuracy: 55.00%
Epoch 4, loss 30.76951747378161, train accuracy: 56.44%
Validation accuracy: 57.00%
Best Valid accuracy: 57.00%
Epoch 5, loss 30.645790379550643, train accuracy: 54.89%
Validation accuracy: 61.00%
Best Valid accuracy: 61.00%
Epoch 6, loss 30.132077015628692, train accuracy: 59.11%
Validation accuracy: 48.00%
Best Valid accuracy: 61.00%
Epoch 7, loss 29.99047051293503, train accuracy: 62.00%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 8, loss 29.649171203758847, train accuracy: 63.11%
Validation accuracy: 58.00%
Best Valid accuracy: 69.00%
Epoch 9, loss 29.423283323328068, train accuracy: 64.44%
Validation accuracy: 60.00%
Best Valid accuracy: 69.00%
Epoch 10, loss 28.926315232865793, train accuracy: 65.78%
Validation accuracy: 65.00%
Best Valid accuracy: 69.00%
Epoch 11, loss 28.668020390897777, train accuracy: 66.00%
Validation accuracy: 67.00%
Best Valid accuracy: 69.00%
Epoch 12, loss 28.206769256926698, train accuracy: 67.78%
Validation accuracy: 65.00%
Best Valid accuracy: 69.00%
Epoch 13, loss 27.532747422695827, train accuracy: 70.44%
Validation accuracy: 66.00%
Best Valid accuracy: 69.00%
Epoch 14, loss 26.882981685257892, train accuracy: 72.00%
Validation accuracy: 62.00%
Best Valid accuracy: 69.00%
Epoch 15, loss 26.66811993683997, train accuracy: 70.67%
Validation accuracy: 67.00%
Best Valid accuracy: 69.00%
Epoch 16, loss 25.70882506298283, train accuracy: 74.22%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 17, loss 25.481190433393905, train accuracy: 70.22%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 18, loss 24.051822281075623, train accuracy: 74.22%
Validation accuracy: 64.00%
Best Valid accuracy: 70.00%
Epoch 19, loss 23.84369452931552, train accuracy: 74.67%
Validation accuracy: 60.00%
Best Valid accuracy: 70.00%
Epoch 20, loss 22.962443872603178, train accuracy: 76.89%
Validation accuracy: 67.00%
Best Valid accuracy: 70.00%
Epoch 21, loss 22.85244077556335, train accuracy: 75.56%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 22, loss 22.836463431333367, train accuracy: 76.89%
Validation accuracy: 65.00%
Best Valid accuracy: 73.00%
Epoch 23, loss 21.206579502922565, train accuracy: 78.89%
Validation accuracy: 71.00%
Best Valid accuracy: 73.00%
Epoch 24, loss 20.602810729209523, train accuracy: 80.22%
Validation accuracy: 70.00%
Best Valid accuracy: 73.00%
Epoch 25, loss 20.357364944441837, train accuracy: 78.89%
Validation accuracy: 66.00%
Best Valid accuracy: 73.00%
```

Log for run_mnist (epoch=25, learning rate=0.01):
```
note: for convenience of length, epoch 1-19 log is ommitted
Epoch 20 loss 0.007317099665147264 valid acc 14/16
Epoch 20 loss 0.4031872707325314 valid acc 14/16
Epoch 20 loss 0.8517585253534691 valid acc 15/16
Epoch 20 loss 0.44624778431063766 valid acc 14/16
Epoch 20 loss 0.10419234639265784 valid acc 14/16
Epoch 20 loss 0.47058719113143 valid acc 14/16
Epoch 20 loss 0.9254753593936951 valid acc 15/16
Epoch 20 loss 0.6416452271871501 valid acc 15/16
Epoch 20 loss 0.4721065399567794 valid acc 15/16
Epoch 20 loss 0.14858332045193914 valid acc 15/16
Epoch 20 loss 0.42569600713897104 valid acc 15/16
Epoch 20 loss 0.7818387926546586 valid acc 15/16
Epoch 20 loss 0.996756612665848 valid acc 15/16
Epoch 20 loss 0.8824491821222644 valid acc 15/16
Epoch 20 loss 0.6743628079323061 valid acc 15/16
Epoch 20 loss 0.5064006526633189 valid acc 15/16
Epoch 20 loss 0.7066614449190942 valid acc 16/16
Epoch 20 loss 0.6291487382373777 valid acc 16/16
Epoch 20 loss 0.4114094710506858 valid acc 16/16
Epoch 20 loss 0.5799562165090216 valid acc 15/16
Epoch 20 loss 0.9908982529308479 valid acc 14/16
Epoch 20 loss 0.2939721439993759 valid acc 15/16
Epoch 20 loss 0.6084177502999644 valid acc 16/16
Epoch 20 loss 0.165947739225143 valid acc 16/16
Epoch 20 loss 0.19764823579017277 valid acc 15/16
Epoch 20 loss 1.0761279234167782 valid acc 16/16
Epoch 20 loss 0.28568533614780983 valid acc 16/16
Epoch 20 loss 0.20371790443937282 valid acc 15/16
Epoch 20 loss 0.5042324239782903 valid acc 15/16
Epoch 20 loss 0.1622375096349032 valid acc 14/16
Epoch 20 loss 0.3751709132301205 valid acc 15/16
Epoch 20 loss 0.28034346541677835 valid acc 16/16
Epoch 20 loss 0.07217446741438183 valid acc 16/16
Epoch 20 loss 0.5082991395918567 valid acc 16/16
Epoch 20 loss 1.2675497529989812 valid acc 15/16
Epoch 20 loss 0.3005822165893083 valid acc 16/16
Epoch 20 loss 0.2835864543614907 valid acc 14/16
Epoch 20 loss 0.8495970569390936 valid acc 15/16
Epoch 20 loss 0.4295126202804893 valid acc 14/16
Epoch 20 loss 0.27399713003969833 valid acc 15/16
Epoch 20 loss 0.14221616247975066 valid acc 14/16
Epoch 20 loss 0.28689985635022797 valid acc 14/16
Epoch 20 loss 0.574765010079733 valid acc 16/16
Epoch 20 loss 0.282134124803268 valid acc 14/16
Epoch 20 loss 1.0085096062199634 valid acc 14/16
Epoch 20 loss 0.2914668779900925 valid acc 14/16
Epoch 20 loss 0.8127661869305015 valid acc 14/16
Epoch 20 loss 0.7181204177613949 valid acc 14/16
Epoch 20 loss 0.18753961605868846 valid acc 14/16
Epoch 20 loss 0.21353958890439817 valid acc 14/16
Epoch 20 loss 0.40908508206220706 valid acc 14/16
Epoch 20 loss 0.42725108672387313 valid acc 14/16
Epoch 20 loss 0.4544695402920276 valid acc 14/16
Epoch 20 loss 0.2513742528840748 valid acc 14/16
Epoch 20 loss 0.5914706646453941 valid acc 15/16
Epoch 20 loss 0.0864711832785382 valid acc 15/16
Epoch 20 loss 0.5621988282378223 valid acc 14/16
Epoch 20 loss 0.3354898936625621 valid acc 15/16
Epoch 20 loss 0.46930422384549264 valid acc 15/16
Epoch 20 loss 0.29575281100468476 valid acc 14/16
Epoch 20 loss 0.2617893454128029 valid acc 15/16
Epoch 20 loss 0.29159559738715546 valid acc 15/16
Epoch 20 loss 0.6372542139001255 valid acc 16/16
Epoch 21 loss 0.008562344403778566 valid acc 15/16
Epoch 21 loss 0.21884986035044232 valid acc 14/16
Epoch 21 loss 0.474830232065949 valid acc 14/16
Epoch 21 loss 0.5355442634153659 valid acc 14/16
Epoch 21 loss 0.6108538424126276 valid acc 15/16
Epoch 21 loss 0.6881402277600894 valid acc 16/16
Epoch 21 loss 0.40356184788035987 valid acc 16/16
Epoch 21 loss 0.988573972618416 valid acc 15/16
Epoch 21 loss 0.2039804230762276 valid acc 15/16
Epoch 21 loss 0.15409379894932576 valid acc 15/16
Epoch 21 loss 0.40117251989238006 valid acc 14/16
Epoch 21 loss 0.5820167391899944 valid acc 15/16
Epoch 21 loss 0.5484607684764369 valid acc 15/16
Epoch 21 loss 0.32960020958122954 valid acc 15/16
Epoch 21 loss 0.623766252453233 valid acc 16/16
Epoch 21 loss 0.17347334886764648 valid acc 15/16
Epoch 21 loss 0.585697567109652 valid acc 15/16
Epoch 21 loss 0.5567027891206466 valid acc 15/16
Epoch 21 loss 0.6210604490280955 valid acc 15/16
Epoch 21 loss 0.44771258550212106 valid acc 15/16
Epoch 21 loss 1.1515348230059912 valid acc 15/16
Epoch 21 loss 0.23991945064560766 valid acc 15/16
Epoch 21 loss 0.10727174738669112 valid acc 15/16
Epoch 21 loss 1.097954507281165 valid acc 16/16
Epoch 21 loss 0.3508920969325114 valid acc 15/16
Epoch 21 loss 0.6540382517327471 valid acc 15/16
Epoch 21 loss 0.3588192956594062 valid acc 15/16
Epoch 21 loss 0.44380241378718727 valid acc 14/16
Epoch 21 loss 0.3483457811038832 valid acc 15/16
Epoch 21 loss 0.09402605408886283 valid acc 15/16
Epoch 21 loss 0.1782966198087953 valid acc 15/16
Epoch 21 loss 0.36970826610559027 valid acc 15/16
Epoch 21 loss 0.2618690943056055 valid acc 15/16
Epoch 21 loss 0.41504994744402324 valid acc 16/16
Epoch 21 loss 1.0248696889643365 valid acc 16/16
Epoch 21 loss 0.28177543839431207 valid acc 16/16
Epoch 21 loss 0.18408601103212285 valid acc 16/16
Epoch 21 loss 0.24100784487225452 valid acc 16/16
Epoch 21 loss 0.7792152491338518 valid acc 15/16
Epoch 21 loss 0.6113796706054057 valid acc 15/16
Epoch 21 loss 0.11477206474927493 valid acc 15/16
Epoch 21 loss 0.7506279550941083 valid acc 14/16
Epoch 21 loss 0.72108781302557 valid acc 15/16
Epoch 21 loss 0.08783310028363037 valid acc 15/16
Epoch 21 loss 0.3092300905729134 valid acc 15/16
Epoch 21 loss 0.1736032111007892 valid acc 15/16
Epoch 21 loss 0.516405493426118 valid acc 15/16
Epoch 21 loss 0.8183112240790509 valid acc 15/16
Epoch 21 loss 0.30054157748843624 valid acc 15/16
Epoch 21 loss 0.9112129347477023 valid acc 15/16
Epoch 21 loss 0.5577919425196569 valid acc 15/16
Epoch 21 loss 0.7662164169433252 valid acc 15/16
Epoch 21 loss 0.3616735502575158 valid acc 15/16
Epoch 21 loss 0.17780323634870104 valid acc 14/16
Epoch 21 loss 0.5086656565303298 valid acc 15/16
Epoch 21 loss 0.1220103012518407 valid acc 14/16
Epoch 21 loss 0.41227935623639334 valid acc 14/16
Epoch 21 loss 0.43004866425955934 valid acc 16/16
Epoch 21 loss 0.5147426372160512 valid acc 15/16
Epoch 21 loss 0.3595490730441817 valid acc 15/16
Epoch 21 loss 0.33102695805676163 valid acc 15/16
Epoch 21 loss 0.36385638051390623 valid acc 15/16
Epoch 21 loss 0.5402624487300822 valid acc 15/16
Epoch 22 loss 0.01986500954652287 valid acc 15/16
Epoch 22 loss 0.21945472701216068 valid acc 15/16
Epoch 22 loss 0.7718866303645623 valid acc 14/16
Epoch 22 loss 0.13904256244142704 valid acc 14/16
Epoch 22 loss 0.28289726364847406 valid acc 14/16
Epoch 22 loss 0.6973198259196935 valid acc 14/16
Epoch 22 loss 0.43835174081327344 valid acc 15/16
Epoch 22 loss 0.22974593713610403 valid acc 15/16
Epoch 22 loss 0.4357663901297512 valid acc 14/16
Epoch 22 loss 0.2215019907017276 valid acc 15/16
Epoch 22 loss 0.2751504570357502 valid acc 15/16
Epoch 22 loss 0.22153173428569695 valid acc 15/16
Epoch 22 loss 0.5275229785564508 valid acc 15/16
Epoch 22 loss 0.4132575312970981 valid acc 15/16
Epoch 22 loss 0.7687133971153303 valid acc 15/16
Epoch 22 loss 0.48349052763410405 valid acc 16/16
Epoch 22 loss 0.473613039346881 valid acc 16/16
Epoch 22 loss 0.2981115402654513 valid acc 16/16
Epoch 22 loss 0.3558834587413533 valid acc 16/16
Epoch 22 loss 0.31794297184015397 valid acc 14/16
Epoch 22 loss 0.5157063048441433 valid acc 15/16
Epoch 22 loss 0.46910074450951234 valid acc 14/16
Epoch 22 loss 0.38653228540416157 valid acc 14/16
Epoch 22 loss 0.3010318770053329 valid acc 14/16
Epoch 22 loss 0.4632451824387945 valid acc 15/16
Epoch 22 loss 0.6110708332511531 valid acc 15/16
Epoch 22 loss 0.24386868903608994 valid acc 15/16
Epoch 22 loss 0.13239815057158572 valid acc 15/16
Epoch 22 loss 0.22763222442103737 valid acc 14/16
Epoch 22 loss 0.10055302684862133 valid acc 14/16
Epoch 22 loss 0.46182231514406435 valid acc 15/16
Epoch 22 loss 0.11952783640787834 valid acc 14/16
Epoch 22 loss 0.25913260897811313 valid acc 15/16
Epoch 22 loss 0.5200318144545466 valid acc 14/16
Epoch 22 loss 0.7460354416399633 valid acc 15/16
Epoch 22 loss 0.4220069938641045 valid acc 15/16
Epoch 22 loss 0.42526954216938617 valid acc 15/16
Epoch 22 loss 0.13553627826583997 valid acc 15/16
Epoch 22 loss 0.19935828232894925 valid acc 14/16
Epoch 22 loss 0.14178422045917377 valid acc 15/16
Epoch 22 loss 0.2683160137696885 valid acc 15/16
Epoch 22 loss 0.1437876185463496 valid acc 15/16
Epoch 22 loss 0.26275081130900924 valid acc 15/16
Epoch 22 loss 0.1478689114043572 valid acc 14/16
Epoch 22 loss 0.368191515815225 valid acc 15/16
Epoch 22 loss 0.03807896729313587 valid acc 15/16
Epoch 22 loss 0.4235672405194585 valid acc 15/16
Epoch 22 loss 0.45032640753936365 valid acc 15/16
Epoch 22 loss 0.2298305686335806 valid acc 15/16
Epoch 22 loss 0.20181098693833316 valid acc 15/16
Epoch 22 loss 0.24105769769186552 valid acc 15/16
Epoch 22 loss 0.35695250610068346 valid acc 15/16
Epoch 22 loss 0.3480988400513874 valid acc 15/16
Epoch 22 loss 0.38107303242988366 valid acc 15/16
Epoch 22 loss 0.7338446105598839 valid acc 15/16
Epoch 22 loss 0.12432640272460782 valid acc 15/16
Epoch 22 loss 0.2653542157478852 valid acc 15/16
Epoch 22 loss 0.1543093963610812 valid acc 15/16
Epoch 22 loss 0.5953655809399315 valid acc 14/16
Epoch 22 loss 0.17352416343430957 valid acc 14/16
Epoch 22 loss 0.2887390530467471 valid acc 14/16
Epoch 22 loss 0.1724408267027668 valid acc 14/16
Epoch 22 loss 0.3251628508127979 valid acc 14/16
Epoch 23 loss 0.0027953533362696237 valid acc 14/16
Epoch 23 loss 0.2926286814688016 valid acc 14/16
Epoch 23 loss 0.9163602369540884 valid acc 15/16
Epoch 23 loss 0.30171144446337134 valid acc 15/16
Epoch 23 loss 0.14463244786791557 valid acc 15/16
Epoch 23 loss 0.08005643336780444 valid acc 15/16
Epoch 23 loss 0.7342984548100476 valid acc 14/16
Epoch 23 loss 0.6332269322619462 valid acc 16/16
Epoch 23 loss 0.6494278095439694 valid acc 15/16
Epoch 23 loss 0.10522459428060937 valid acc 14/16
Epoch 23 loss 0.24103568573396428 valid acc 15/16
Epoch 23 loss 0.44871923462622115 valid acc 15/16
Epoch 23 loss 0.3686555259390557 valid acc 13/16
Epoch 23 loss 0.36523836111483354 valid acc 14/16
Epoch 23 loss 0.3117880968161276 valid acc 15/16
Epoch 23 loss 0.595223993553504 valid acc 15/16
Epoch 23 loss 0.9726101065937409 valid acc 16/16
Epoch 23 loss 0.7364383721656997 valid acc 16/16
Epoch 23 loss 0.3582051622842492 valid acc 16/16
Epoch 23 loss 0.4819358454822407 valid acc 15/16
Epoch 23 loss 0.9576045885920828 valid acc 15/16
Epoch 23 loss 0.20321718084810778 valid acc 16/16
Epoch 23 loss 0.35232411484121284 valid acc 16/16
Epoch 23 loss 0.33709843939878403 valid acc 16/16
Epoch 23 loss 0.11747973496411607 valid acc 16/16
Epoch 23 loss 0.29685052084244834 valid acc 14/16
Epoch 23 loss 0.1933273682229506 valid acc 15/16
Epoch 23 loss 0.27622978819721056 valid acc 15/16
Epoch 23 loss 0.35253668177254904 valid acc 15/16
Epoch 23 loss 0.14572007407579626 valid acc 15/16
Epoch 23 loss 0.27884131174950355 valid acc 15/16
Epoch 23 loss 0.3112791788704493 valid acc 15/16
Epoch 23 loss 0.3425626930601486 valid acc 15/16
Epoch 23 loss 0.14930598757100458 valid acc 15/16
Epoch 23 loss 0.6492622590739009 valid acc 15/16
Epoch 23 loss 0.7572974967328261 valid acc 15/16
Epoch 23 loss 0.507019950261127 valid acc 15/16
Epoch 23 loss 0.22069823868930716 valid acc 14/16
Epoch 23 loss 0.09449895242079953 valid acc 14/16
Epoch 23 loss 0.45705061524090573 valid acc 14/16
Epoch 23 loss 0.08224255057564572 valid acc 14/16
Epoch 23 loss 0.47047026204557096 valid acc 15/16
Epoch 23 loss 0.47099944470640775 valid acc 15/16
Epoch 23 loss 0.18797484869009462 valid acc 15/16
Epoch 23 loss 0.4427973766737958 valid acc 15/16
Epoch 23 loss 0.06359126616423871 valid acc 15/16
Epoch 23 loss 0.28702683290569914 valid acc 16/16
Epoch 23 loss 1.0525968090673639 valid acc 15/16
Epoch 23 loss 0.08951335647160641 valid acc 15/16
Epoch 23 loss 0.4187317317803792 valid acc 15/16
Epoch 23 loss 0.2793931571000352 valid acc 15/16
Epoch 23 loss 0.321834311682788 valid acc 15/16
Epoch 23 loss 0.07938048419138954 valid acc 15/16
Epoch 23 loss 0.07116567057940748 valid acc 15/16
Epoch 23 loss 0.7496489214301976 valid acc 15/16
Epoch 23 loss 0.06969576954566359 valid acc 15/16
Epoch 23 loss 0.3255471449955777 valid acc 15/16
Epoch 23 loss 0.28757211797361837 valid acc 15/16
Epoch 23 loss 0.4205332709731221 valid acc 15/16
Epoch 23 loss 0.3593775232925412 valid acc 14/16
Epoch 23 loss 0.08030996950929442 valid acc 14/16
Epoch 23 loss 0.15285274611375615 valid acc 15/16
Epoch 23 loss 0.7474509874190347 valid acc 15/16
Epoch 24 loss 0.0007433871347275467 valid acc 15/16
Epoch 24 loss 0.25967476139808876 valid acc 15/16
Epoch 24 loss 0.627454595187823 valid acc 15/16
Epoch 24 loss 0.3078897051876133 valid acc 15/16
Epoch 24 loss 0.16004583223102004 valid acc 16/16
Epoch 24 loss 0.4219975811753729 valid acc 15/16
Epoch 24 loss 0.9806794605362101 valid acc 15/16
Epoch 24 loss 0.5253234116084294 valid acc 15/16
Epoch 24 loss 0.20756855587457207 valid acc 15/16
Epoch 24 loss 0.2081576780684528 valid acc 15/16
Epoch 24 loss 0.27339501929458165 valid acc 15/16
Epoch 24 loss 0.6552351623534598 valid acc 15/16
Epoch 24 loss 0.33587930568938634 valid acc 15/16
Epoch 24 loss 0.23391354014096888 valid acc 15/16
Epoch 24 loss 0.712130219996691 valid acc 15/16
Epoch 24 loss 0.33136623247598274 valid acc 14/16
Epoch 24 loss 0.8968527022127895 valid acc 16/16
Epoch 24 loss 0.53148127750604 valid acc 16/16
Epoch 24 loss 0.23711441179716286 valid acc 16/16
Epoch 24 loss 0.8145078277147668 valid acc 16/16
Epoch 24 loss 0.6299550422904165 valid acc 16/16
Epoch 24 loss 0.1993052279519567 valid acc 16/16
Epoch 24 loss 0.33807165657806665 valid acc 15/16
Epoch 24 loss 0.5355106413844923 valid acc 15/16
Epoch 24 loss 0.39324165008289363 valid acc 15/16
Epoch 24 loss 0.44246676331860585 valid acc 15/16
Epoch 24 loss 0.37946826065273265 valid acc 15/16
Epoch 24 loss 0.4095385890468219 valid acc 15/16
Epoch 24 loss 0.6382962614997727 valid acc 14/16
Epoch 24 loss 0.16104853386317775 valid acc 15/16
Epoch 24 loss 0.13625399687577222 valid acc 15/16
Epoch 24 loss 0.6787848373802554 valid acc 14/16
Epoch 24 loss 0.2515832170164629 valid acc 15/16
Epoch 24 loss 0.5196167365610196 valid acc 15/16
Epoch 24 loss 0.8086436512553491 valid acc 15/16
Epoch 24 loss 0.4488111418241986 valid acc 16/16
Epoch 24 loss 0.7104445282722696 valid acc 15/16
Epoch 24 loss 0.2137622963633068 valid acc 15/16
Epoch 24 loss 0.8444216571948149 valid acc 15/16
Epoch 24 loss 0.4703800382780709 valid acc 15/16
Epoch 24 loss 0.23576575926560495 valid acc 15/16
Epoch 24 loss 0.1941078111843251 valid acc 15/16
Epoch 24 loss 0.2484747930540691 valid acc 15/16
Epoch 24 loss 0.12457473565316929 valid acc 15/16
Epoch 24 loss 0.3490102484812114 valid acc 15/16
Epoch 24 loss 0.19910821889370192 valid acc 15/16
Epoch 24 loss 0.5618828146219774 valid acc 15/16
Epoch 24 loss 0.4489355635039542 valid acc 15/16
Epoch 24 loss 0.49111917694736057 valid acc 15/16
Epoch 24 loss 0.21230935039577187 valid acc 15/16
Epoch 24 loss 0.19521320705560935 valid acc 15/16
Epoch 24 loss 0.4221182679480174 valid acc 15/16
Epoch 24 loss 0.09186008019084702 valid acc 15/16
Epoch 24 loss 0.09119706008573725 valid acc 15/16
Epoch 24 loss 0.3466066057733437 valid acc 15/16
Epoch 24 loss 0.055234886537324834 valid acc 16/16
Epoch 24 loss 0.15127166380283297 valid acc 16/16
Epoch 24 loss 0.40024800217608647 valid acc 16/16
Epoch 24 loss 0.5192478830971762 valid acc 16/16
Epoch 24 loss 0.25965532556312254 valid acc 15/16
Epoch 24 loss 0.39523631907090856 valid acc 14/16
Epoch 24 loss 0.6856951223612964 valid acc 15/16
Epoch 24 loss 0.4261847297670502 valid acc 15/16
Epoch 25 loss 0.0037585219938993664 valid acc 15/16
Epoch 25 loss 0.4934507988822062 valid acc 15/16
Epoch 25 loss 0.737945244098182 valid acc 14/16
Epoch 25 loss 0.10176227539050953 valid acc 14/16
Epoch 25 loss 0.07468697959197729 valid acc 14/16
Epoch 25 loss 0.4138780446894844 valid acc 15/16
Epoch 25 loss 0.4696308305058174 valid acc 15/16
Epoch 25 loss 0.5458114681686597 valid acc 15/16
Epoch 25 loss 0.18392701704495085 valid acc 15/16
Epoch 25 loss 0.4040949792745633 valid acc 15/16
Epoch 25 loss 0.18235966477957047 valid acc 15/16
Epoch 25 loss 0.48975303021289385 valid acc 15/16
Epoch 25 loss 0.36764999277444277 valid acc 15/16
Epoch 25 loss 0.4456980759739908 valid acc 15/16
Epoch 25 loss 0.8362064303206096 valid acc 15/16
Epoch 25 loss 0.4491022376333913 valid acc 15/16
Epoch 25 loss 0.6955283173599325 valid acc 15/16
Epoch 25 loss 0.9764393688308465 valid acc 16/16
Epoch 25 loss 0.19655419032516247 valid acc 16/16
Epoch 25 loss 0.2513819357021895 valid acc 16/16
Epoch 25 loss 0.7789674670351093 valid acc 15/16
Epoch 25 loss 0.4867368781174315 valid acc 16/16
Epoch 25 loss 0.42883303996456346 valid acc 15/16
Epoch 25 loss 0.8081145673053507 valid acc 16/16
Epoch 25 loss 0.21861384518088375 valid acc 15/16
Epoch 25 loss 0.23792628188922435 valid acc 16/16
Epoch 25 loss 0.40647897858408416 valid acc 16/16
Epoch 25 loss 0.11931626827435945 valid acc 16/16
Epoch 25 loss 0.5995772124652166 valid acc 15/16
Epoch 25 loss 0.21885216547503716 valid acc 15/16
Epoch 25 loss 0.2443822101268343 valid acc 15/16
Epoch 25 loss 0.2609435900959528 valid acc 15/16
Epoch 25 loss 0.3755462422663655 valid acc 15/16
Epoch 25 loss 0.882368330002052 valid acc 16/16
Epoch 25 loss 1.1939095949902025 valid acc 16/16
Epoch 25 loss 0.35201506540585287 valid acc 16/16
Epoch 25 loss 0.1306865886520588 valid acc 15/16
Epoch 25 loss 0.21143532627421813 valid acc 16/16
Epoch 25 loss 0.29586685505054267 valid acc 15/16
Epoch 25 loss 0.1457856730226706 valid acc 16/16
Epoch 25 loss 0.05741788670177422 valid acc 16/16
Epoch 25 loss 0.23836880752442632 valid acc 16/16
Epoch 25 loss 0.25952996009252854 valid acc 16/16
Epoch 25 loss 0.2740361214476949 valid acc 15/16
Epoch 25 loss 0.44280160934181123 valid acc 15/16
Epoch 25 loss 0.7702991463775415 valid acc 14/16
Epoch 25 loss 0.5902206493328727 valid acc 14/16
Epoch 25 loss 0.43114792113828293 valid acc 14/16
Epoch 25 loss 0.42719482688091154 valid acc 15/16
Epoch 25 loss 0.15179549213931842 valid acc 15/16
Epoch 25 loss 0.48029367496104336 valid acc 15/16
Epoch 25 loss 0.2422666440665501 valid acc 15/16
Epoch 25 loss 0.28814746310259187 valid acc 15/16
Epoch 25 loss 0.452642838678754 valid acc 15/16
Epoch 25 loss 0.061437975260257016 valid acc 15/16
Epoch 25 loss 0.5657137893700326 valid acc 15/16
Epoch 25 loss 0.2783672266053401 valid acc 15/16
Epoch 25 loss 0.6174306835019703 valid acc 15/16
Epoch 25 loss 0.33510028472304393 valid acc 14/16
Epoch 25 loss 0.12822067615311517 valid acc 15/16
Epoch 25 loss 0.40919602700972746 valid acc 15/16
Epoch 25 loss 0.3980077288071602 valid acc 16/16
```