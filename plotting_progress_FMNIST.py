import csv

import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

# Typing
import typing
from typing import TypeVar, Generic
from collections.abc import Callable

from tqdm import tqdm
from collections import namedtuple
import statistics
import dataclasses
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

from matplotlib.backends.backend_pdf import PdfPages

# cd Documents/
# cd populationDescent/
# python3 -m venv ~/venv-metal
# source ~/venv-metal/bin/activate
# python3 -m plotting_progress_FMNIST




## ESGD
ema_ESGD = [0.7147568140029907, 0.7147568140029907, 0.7063239600372315, 0.6904548344116211, 0.6737380081016541, 0.6529779560326385, 0.6312089610408707, 0.6110588607684639, 0.5913711296059967, 0.572636624606098, 0.5552297108408617, 0.5392673656436927, 0.5238285007548148, 0.5087118442230059, 0.4956699083101123, 0.48244700060707657, 0.46923594914217887, 0.45864946948713153, 0.44770697970890483, 0.4363916123006059, 0.4253704713428003, 0.4155497089222654, 0.4061756981702671, 0.3984000114117884, 0.38980988669826, 0.3817943898096688, 0.3738245924259729, 0.36680000767697457, 0.36022366800894945, 0.35444774962476444, 0.3475625654454072, 0.3411030203457212, 0.3351495013614478, 0.3299197999149545, 0.3247554833701239, 0.32005707379558557, 0.31582810273270795, 0.3117660646018474, 0.30849209666404925, 0.30568624790806, 0.3026341200616815, 0.29921485983135027, 0.2960857052815114, 0.2940475748803158, 0.29156016368859466, 0.28966137870544306, 0.2875035294758747, 0.28509831545479325, 0.28341998261088636, 0.2811936787081466, 0.2796161113505995, 0.27853373191643677, 0.2782309327044093, 0.276636747925103, 0.2746382516222778, 0.27277284675576535, 0.27078584412221535, 0.269158627746264, 0.2677651829413413, 0.26762456534729945, 0.26611578773999106, 0.2649905692608108, 0.26415705728807604, 0.26383652404080526, 0.26298465265688925, 0.26200204312316444, 0.261466875126766, 0.26094804889453066, 0.2600120017346079, 0.2588758463462891, 0.25870860681599694, 0.2582846699434488, 0.2589567838988872, 0.2585021208717381, 0.2577046227068265, 0.25628141042088504, 0.2563630747600946, 0.2564744426790971, 0.25531082490063595, 0.25657403019228775, 0.25644145208322283, 0.25716294991365024, 0.2566142325609113, 0.25601394599877464, 0.25603160533728603, 0.25647128072103026, 0.25580476763534693, 0.25641680803079736, 0.255119834597522, 0.2555565920921689, 0.25502196578614644, 0.2548081960526055, 0.25487950303735846, 0.2541135944382773]
x_ESGD = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500, 44000, 44500, 45000, 45500, 46000, 46500]
history_ESGD = [0.7147568140029907, 0.6304282743453979, 0.5476327037811279, 0.5232865713119507, 0.466137487411499, 0.4352880061149597, 0.429707958316803, 0.4141815491437912, 0.40402607960700987, 0.39856748695373534, 0.39560625886917117, 0.3848787167549133, 0.3726619354367256, 0.37829248509407043, 0.36344083127975463, 0.35033648595809935, 0.3633711525917053, 0.3492245717048645, 0.3345533056259155, 0.32618020272254944, 0.3271628471374512, 0.3218096014022827, 0.3284188305854797, 0.3124987642765045, 0.3096549178123474, 0.30209641597270964, 0.30357874493598935, 0.3010366109967232, 0.302464484167099, 0.28559590783119204, 0.2829671144485474, 0.2815678305029869, 0.2828524868965149, 0.2782766344666481, 0.2777713876247406, 0.27776736316680906, 0.27520772142410277, 0.2790263852238655, 0.2804336091041565, 0.2751649694442749, 0.26844151775836944, 0.267923314332962, 0.2757044012695551, 0.26917346296310424, 0.27257231385707853, 0.2680828864097595, 0.2634513892650604, 0.2683149870157242, 0.26115694358348845, 0.26541800513267516, 0.26879231700897216, 0.27550573979616166, 0.26228908491134645, 0.2566517848968506, 0.2559842029571533, 0.2529028204202652, 0.2545136803627014, 0.25522417969703676, 0.26635900700092313, 0.2525367892742157, 0.25486360294818877, 0.25665544953346253, 0.26095172481536866, 0.2553178102016449, 0.2531585573196411, 0.25665036315917966, 0.25627861280441283, 0.25158757729530334, 0.24865044785141946, 0.2572034510433674, 0.25446923809051514, 0.26500580949783326, 0.2544101536273956, 0.25052713922262193, 0.24347249984741212, 0.2570980538129807, 0.25747675395011904, 0.24483826489448549, 0.2679428778171539, 0.2552482491016388, 0.2636564303874969, 0.251675776386261, 0.25061136693954467, 0.25619053938388825, 0.2604283591747284, 0.24980614986419677, 0.2619251715898514, 0.24344707369804383, 0.2594874095439911, 0.2502103290319443, 0.252884268450737, 0.255521265900135, 0.24722041704654693]

ema_ESGD15 = [0.7147568140029907, 0.7147568140029907, 0.7063239600372315, 0.6904548344116211, 0.6737380081016541, 0.6529779560326385, 0.6312089610408707, 0.6110588607684639, 0.5913711296059967, 0.572636624606098, 0.5552297108408617, 0.5392673656436927, 0.5238285007548148, 0.5087118442230059, 0.4956699083101123, 0.48244700060707657, 0.46923594914217887, 0.45864946948713153, 0.44770697970890483, 0.4363916123006059, 0.4253704713428003, 0.4155497089222654, 0.4061756981702671, 0.3984000114117884, 0.38980988669826, 0.3817943898096688, 0.3738245924259729, 0.36680000767697457, 0.36022366800894945, 0.35444774962476444, 0.3475625654454072, 0.3411030203457212, 0.3351495013614478, 0.3299197999149545, 0.3247554833701239, 0.32005707379558557, 0.31582810273270795, 0.3117660646018474, 0.30849209666404925, 0.30568624790806, 0.3026341200616815, 0.29921485983135027, 0.2960857052815114, 0.2940475748803158, 0.29156016368859466, 0.28966137870544306, 0.2875035294758747, 0.28509831545479325, 0.28341998261088636, 0.2811936787081466, 0.2796161113505995, 0.27853373191643677, 0.2782309327044093, 0.276636747925103, 0.2746382516222778, 0.27277284675576535, 0.27078584412221535, 0.269158627746264, 0.2677651829413413, 0.26762456534729945, 0.26611578773999106, 0.2649905692608108, 0.26415705728807604, 0.26383652404080526, 0.26298465265688925, 0.26200204312316444, 0.261466875126766, 0.26094804889453066, 0.2600120017346079, 0.2588758463462891, 0.25870860681599694, 0.2582846699434488, 0.2589567838988872, 0.2585021208717381, 0.2577046227068265, 0.25628141042088504, 0.2563630747600946, 0.2564744426790971, 0.25531082490063595, 0.25657403019228775, 0.25644145208322283, 0.25716294991365024, 0.2566142325609113, 0.25601394599877464, 0.25603160533728603, 0.25647128072103026, 0.25580476763534693, 0.25641680803079736, 0.255119834597522, 0.2555565920921689, 0.25502196578614644, 0.2548081960526055, 0.25487950303735846, 0.2541135944382773, 0.25360400719312665, 0.25336073130310627, 0.2525886586863493, 0.2526234178711797, 0.2547469616851054, 0.2543839726609745, 0.25512860971952306, 0.25458916766045836, 0.25470061674772215, 0.25463990735285147, 0.255846604698354, 0.2564842093886024, 0.25657448239991837, 0.2569123917377273, 0.2573832137736332, 0.2581938688152419, 0.25931252904743435, 0.25923952245662046, 0.25893585069611164, 0.2584748849931463, 0.2584275877746735, 0.25837928839976737, 0.259971761379735, 0.2601907520479825, 0.26102391192318797, 0.26074930711926636, 0.2596799114806201, 0.25958216522181743, 0.2589779667488758, 0.26051912302780367, 0.26099575766391947, 0.26154166156471426, 0.26168427492637025, 0.26228143535249165, 0.2634330910139626, 0.26415685951113815, 0.2642197016137849, 0.26527442789357997, 0.26713236922277894, 0.26750524124074476, 0.2682328776148125, 0.2706172858500926, 0.2720228062532349, 0.27360334275561604, 0.2730110239159885, 0.2723750535796723, 0.2719499425855852]
x_ESGD15 = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500, 44000, 44500, 45000, 45500, 46000, 46500, 47000, 47500, 48000, 48500, 49000, 49500, 50000, 50500, 51000, 51500, 52000, 52500, 53000, 53500, 54000, 54500, 55000, 55500, 56000, 56500, 57000, 57500, 58000, 58500, 59000, 59500, 60000, 60500, 61000, 61500, 62000, 62500, 63000, 63500, 64000, 64500, 65000, 65500, 66000, 66500, 67000, 67500, 68000, 68500, 69000, 69500, 70000]
history_ESGD15 = [0.7147568140029907, 0.6304282743453979, 0.5476327037811279, 0.5232865713119507, 0.466137487411499, 0.4352880061149597, 0.429707958316803, 0.4141815491437912, 0.40402607960700987, 0.39856748695373534, 0.39560625886917117, 0.3848787167549133, 0.3726619354367256, 0.37829248509407043, 0.36344083127975463, 0.35033648595809935, 0.3633711525917053, 0.3492245717048645, 0.3345533056259155, 0.32618020272254944, 0.3271628471374512, 0.3218096014022827, 0.3284188305854797, 0.3124987642765045, 0.3096549178123474, 0.30209641597270964, 0.30357874493598935, 0.3010366109967232, 0.302464484167099, 0.28559590783119204, 0.2829671144485474, 0.2815678305029869, 0.2828524868965149, 0.2782766344666481, 0.2777713876247406, 0.27776736316680906, 0.27520772142410277, 0.2790263852238655, 0.2804336091041565, 0.2751649694442749, 0.26844151775836944, 0.267923314332962, 0.2757044012695551, 0.26917346296310424, 0.27257231385707853, 0.2680828864097595, 0.2634513892650604, 0.2683149870157242, 0.26115694358348845, 0.26541800513267516, 0.26879231700897216, 0.27550573979616166, 0.26228908491134645, 0.2566517848968506, 0.2559842029571533, 0.2529028204202652, 0.2545136803627014, 0.25522417969703676, 0.26635900700092313, 0.2525367892742157, 0.25486360294818877, 0.25665544953346253, 0.26095172481536866, 0.2553178102016449, 0.2531585573196411, 0.25665036315917966, 0.25627861280441283, 0.25158757729530334, 0.24865044785141946, 0.2572034510433674, 0.25446923809051514, 0.26500580949783326, 0.2544101536273956, 0.25052713922262193, 0.24347249984741212, 0.2570980538129807, 0.25747675395011904, 0.24483826489448549, 0.2679428778171539, 0.2552482491016388, 0.2636564303874969, 0.251675776386261, 0.25061136693954467, 0.25619053938388825, 0.2604283591747284, 0.24980614986419677, 0.2619251715898514, 0.24344707369804383, 0.2594874095439911, 0.2502103290319443, 0.252884268450737, 0.255521265900135, 0.24722041704654693, 0.24901772198677063, 0.251171248292923, 0.2456400051355362, 0.2529362505346537, 0.273858856010437, 0.25111707144379614, 0.26183034324646, 0.24973418912887574, 0.2557036585330963, 0.25409352279901504, 0.2667068808078766, 0.2622226516008377, 0.2573869395017624, 0.2599535757780075, 0.2616206120967865, 0.26548976418972015, 0.269380471137166, 0.25858246313929556, 0.25620280485153196, 0.25432619366645814, 0.2580019128084183, 0.2579445940256119, 0.27430401819944383, 0.2621616680622101, 0.2685223508000374, 0.25827786388397217, 0.2500553507328033, 0.25870244889259336, 0.2535401804924011, 0.2743895295381546, 0.2652854693889618, 0.2664547966718674, 0.2629677951812744, 0.26765587918758393, 0.2737979919672012, 0.27067077598571776, 0.2647852805376053, 0.27476696441173554, 0.2838538411855698, 0.2708610894024372, 0.2747816049814224, 0.29207695996761324, 0.2846724898815155, 0.2878281712770462, 0.26768015435934067, 0.2666513205528259, 0.2681239436388016]

### These PDs are all with regularization

## PD 50
ema_PD = [0.5350429, 0.5350428819656372, 0.5334884405136109, 0.5248474431037903, 0.5088954495191574, 0.4931205833077431, 0.4807713917601109, 0.4708060048557521, 0.46552393935953384, 0.4488681057348979, 0.4306114076902091, 0.41745608383383925, 0.4070136010703436, 0.3986312502673559, 0.39135356432658075, 0.38401919024999415, 0.3737821646137429, 0.36554510899812515, 0.35732613064435603, 0.34912193171013134, 0.34501218812312545, 0.3411861331759023, 0.3356953118590827, 0.33409125396743256, 0.3282917480689422, 0.32394734987814444, 0.3198931840288569, 0.3150535965467629, 0.31054607133613266, 0.31234867662679766, 0.3104503001912679, 0.30816842730142885, 0.30422361447183677, 0.299455950442866, 0.30066174249403077, 0.29788356107967257, 0.2998127318843945, 0.29406470454556444, 0.2939940398647431, 0.2904959594351688, 0.28449526951992266, 0.2793391289676763, 0.27594864896227755, 0.27995423255523894, 0.28241053323293497, 0.27717322051833365, 0.2757088447240915, 0.2692649413312608, 0.26303551189587715, 0.2610085853262502, 0.2549896685770534]
x_PD = [640, 1280, 1920, 2560, 3200, 3840, 4480, 5120, 5760, 6400, 7040, 7680, 8320, 8960, 9600, 10240, 10880, 11520, 12160, 12800, 13440, 14080, 14720, 15360, 16000, 16640, 17280, 17920, 18560, 19200, 19840, 20480, 21120, 21760, 22400, 23040, 23680, 24320, 24960, 25600, 26240, 26880, 27520, 28160, 28800, 29440, 30080, 30720, 31360, 32000]
history_PD = [0.5350429, 0.51949847, 0.44707847, 0.3653275, 0.3511468, 0.36962867, 0.38111752, 0.41798535, 0.2989656, 0.26630113, 0.29905817, 0.31303126, 0.3231901, 0.3258544, 0.31800982, 0.28164893, 0.2914116, 0.28335533, 0.27528414, 0.3080245, 0.30675164, 0.28627792, 0.31965473, 0.2760962, 0.28484777, 0.2834057, 0.2714973, 0.26997834, 0.32857212, 0.2933649, 0.28763157, 0.2687203, 0.25654697, 0.31151387, 0.27287993, 0.31717527, 0.24233246, 0.29335806, 0.25901324, 0.23048906, 0.23293386, 0.24543433, 0.31600448, 0.30451724, 0.2300374, 0.26252946, 0.21126981, 0.20697065, 0.24276625, 0.20081942]

## PD 65
ema_PD65 = [0.5350429, 0.5350428819656372, 0.5334884405136109, 0.5248474431037903, 0.5088954495191574, 0.4931205833077431, 0.4807713917601109, 0.4708060048557521, 0.46552393935953384, 0.4488681057348979, 0.4306114076902091, 0.41745608383383925, 0.4070136010703436, 0.3986312502673559, 0.39135356432658075, 0.38401919024999415, 0.3737821646137429, 0.36554510899812515, 0.35732613064435603, 0.34912193171013134, 0.34501218812312545, 0.3411861331759023, 0.3356953118590827, 0.33409125396743256, 0.3282917480689422, 0.32394734987814444, 0.3198931840288569, 0.3150535965467629, 0.31054607133613266, 0.31234867662679766, 0.3104503001912679, 0.30816842730142885, 0.30422361447183677, 0.299455950442866, 0.30066174249403077, 0.29788356107967257, 0.2998127318843945, 0.29406470454556444, 0.2939940398647431, 0.2904959594351688, 0.28449526951992266, 0.2793391289676763, 0.27594864896227755, 0.27995423255523894, 0.28241053323293497, 0.27717322051833365, 0.2757088447240915, 0.2692649413312608, 0.26303551189587715, 0.2610085853262502, 0.25758154824290497, 0.2539770808957263, 0.255530691785417, 0.25712801215899167, 0.2529643985437051, 0.25172596539318826, 0.2541830214732313, 0.25834370012939784, 0.2583099546062423, 0.258226383509708, 0.25345056097132496, 0.2507063970603573, 0.24973158320528516, 0.2489560778207384, 0.24398495870233797, 0.24770271883071562]
x_PD65 = [640, 1280, 1920, 2560, 3200, 3840, 4480, 5120, 5760, 6400, 7040, 7680, 8320, 8960, 9600, 10240, 10880, 11520, 12160, 12800, 13440, 14080, 14720, 15360, 16000, 16640, 17280, 17920, 18560, 19200, 19840, 20480, 21120, 21760, 22400, 23040, 23680, 24320, 24960, 25600, 26240, 26880, 27520, 28160, 28800, 29440, 30080, 30720, 31360, 32000, 32640, 33280, 33920, 34560, 35200, 35840, 36480, 37120, 37760, 38400, 39040, 39680, 40320, 40960, 41600]
history_PD65 = [0.5350429, 0.51949847, 0.44707847, 0.3653275, 0.3511468, 0.36962867, 0.38111752, 0.41798535, 0.2989656, 0.26630113, 0.29905817, 0.31303126, 0.3231901, 0.3258544, 0.31800982, 0.28164893, 0.2914116, 0.28335533, 0.27528414, 0.3080245, 0.30675164, 0.28627792, 0.31965473, 0.2760962, 0.28484777, 0.2834057, 0.2714973, 0.26997834, 0.32857212, 0.2933649, 0.28763157, 0.2687203, 0.25654697, 0.31151387, 0.27287993, 0.31717527, 0.24233246, 0.29335806, 0.25901324, 0.23048906, 0.23293386, 0.24543433, 0.31600448, 0.30451724, 0.2300374, 0.26252946, 0.21126981, 0.20697065, 0.24276625, 0.22673821, 0.22153687, 0.2695132, 0.2715039, 0.21549188, 0.24058007, 0.27629653, 0.2957898, 0.25800624, 0.25747424, 0.21046816, 0.22600892, 0.24095826, 0.24197653, 0.19924489, 0.28116256]

##PD 90
ema_PD90 = [0.5350429, 0.5350428819656372, 0.5334884405136109, 0.5248474431037903, 0.5088954495191574, 0.4931205833077431, 0.4807713917601109, 0.4708060048557521, 0.46552393935953384, 0.4488681057348979, 0.4306114076902091, 0.41745608383383925, 0.4070136010703436, 0.3986312502673559, 0.39135356432658075, 0.38401919024999415, 0.3737821646137429, 0.36554510899812515, 0.35732613064435603, 0.34912193171013134, 0.34501218812312545, 0.3411861331759023, 0.3356953118590827, 0.33409125396743256, 0.3282917480689422, 0.32394734987814444, 0.3198931840288569, 0.3150535965467629, 0.31054607133613266, 0.31234867662679766, 0.3104503001912679, 0.30816842730142885, 0.30422361447183677, 0.299455950442866, 0.30066174249403077, 0.29788356107967257, 0.2998127318843945, 0.29406470454556444, 0.2939940398647431, 0.2904959594351688, 0.28449526951992266, 0.2793391289676763, 0.27594864896227755, 0.27995423255523894, 0.28241053323293497, 0.27717322051833365, 0.2757088447240915, 0.2692649413312608, 0.26303551189587715, 0.2610085853262502, 0.25758154824290497, 0.2539770808957263, 0.255530691785417, 0.25712801215899167, 0.2529643985437051, 0.25172596539318826, 0.2541830214732313, 0.25834370012939784, 0.2583099546062423, 0.258226383509708, 0.25345056097132496, 0.2507063970603573, 0.24973158320528516, 0.2489560778207384, 0.24398495870233797, 0.24523666307986877, 0.2464710248022774, 0.24757157616402203, 0.24375806021021748, 0.246408487266245, 0.25289448044235335, 0.25585468061150457, 0.2502087260548593, 0.2519723181662847, 0.2510646570984622, 0.24735875778771343, 0.24462183135786864, 0.24305323050653233, 0.24351380723329152, 0.24625706187735585, 0.24734398882255484, 0.24653545463680204, 0.24611626217900792, 0.24936908319716825, 0.24424766154699185, 0.2422968579972922, 0.24296794685528927, 0.23982078748470176, 0.23882868365221288, 0.24057649203336, 0.23968470807250525]
x_PD90 = [640, 1280, 1920, 2560, 3200, 3840, 4480, 5120, 5760, 6400, 7040, 7680, 8320, 8960, 9600, 10240, 10880, 11520, 12160, 12800, 13440, 14080, 14720, 15360, 16000, 16640, 17280, 17920, 18560, 19200, 19840, 20480, 21120, 21760, 22400, 23040, 23680, 24320, 24960, 25600, 26240, 26880, 27520, 28160, 28800, 29440, 30080, 30720, 31360, 32000, 32640, 33280, 33920, 34560, 35200, 35840, 36480, 37120, 37760, 38400, 39040, 39680, 40320, 40960, 41600, 42240, 42880, 43520, 44160, 44800, 45440, 46080, 46720, 47360, 48000, 48640, 49280, 49920, 50560, 51200, 51840, 52480, 53120, 53760, 54400, 55040, 55680, 56320, 56960, 57600]
history_PD90 = [0.5350429, 0.51949847, 0.44707847, 0.3653275, 0.3511468, 0.36962867, 0.38111752, 0.41798535, 0.2989656, 0.26630113, 0.29905817, 0.31303126, 0.3231901, 0.3258544, 0.31800982, 0.28164893, 0.2914116, 0.28335533, 0.27528414, 0.3080245, 0.30675164, 0.28627792, 0.31965473, 0.2760962, 0.28484777, 0.2834057, 0.2714973, 0.26997834, 0.32857212, 0.2933649, 0.28763157, 0.2687203, 0.25654697, 0.31151387, 0.27287993, 0.31717527, 0.24233246, 0.29335806, 0.25901324, 0.23048906, 0.23293386, 0.24543433, 0.31600448, 0.30451724, 0.2300374, 0.26252946, 0.21126981, 0.20697065, 0.24276625, 0.22673821, 0.22153687, 0.2695132, 0.2715039, 0.21549188, 0.24058007, 0.27629653, 0.2957898, 0.25800624, 0.25747424, 0.21046816, 0.22600892, 0.24095826, 0.24197653, 0.19924489, 0.256502, 0.25758028, 0.25747654, 0.20943642, 0.27026233, 0.31126842, 0.28249648, 0.19939514, 0.26784465, 0.24289571, 0.21400566, 0.2199895, 0.22893582, 0.247659, 0.27094635, 0.25712633, 0.23925865, 0.24234353, 0.27864447, 0.19815487, 0.22473963, 0.24900775, 0.21149635, 0.22989975, 0.25630677, 0.23165865]

ema_PD115 = [0.5350429, 0.5350428819656372, 0.5334884405136109, 0.5248474431037903, 0.5088954495191574, 0.4931205833077431, 0.4807713917601109, 0.4708060048557521, 0.46552393935953384, 0.4488681057348979, 0.4306114076902091, 0.41745608383383925, 0.4070136010703436, 0.3986312502673559, 0.39135356432658075, 0.38401919024999415, 0.3737821646137429, 0.36554510899812515, 0.35732613064435603, 0.34912193171013134, 0.34501218812312545, 0.3411861331759023, 0.3356953118590827, 0.33409125396743256, 0.3282917480689422, 0.32394734987814444, 0.3198931840288569, 0.3150535965467629, 0.31054607133613266, 0.31234867662679766, 0.3104503001912679, 0.30816842730142885, 0.30422361447183677, 0.299455950442866, 0.30066174249403077, 0.29788356107967257, 0.2998127318843945, 0.29406470454556444, 0.2939940398647431, 0.2904959594351688, 0.28449526951992266, 0.2793391289676763, 0.27594864896227755, 0.27995423255523894, 0.28241053323293497, 0.27717322051833365, 0.2757088447240915, 0.2692649413312608, 0.26303551189587715, 0.2610085853262502, 0.25758154824290497, 0.2539770808957263, 0.255530691785417, 0.25712801215899167, 0.2529643985437051, 0.25172596539318826, 0.2541830214732313, 0.25834370012939784, 0.2583099546062423, 0.258226383509708, 0.25345056097132496, 0.2507063970603573, 0.24973158320528516, 0.2489560778207384, 0.24398495870233797, 0.24523666307986877, 0.2464710248022774, 0.24757157616402203, 0.24375806021021748, 0.246408487266245, 0.25289448044235335, 0.25585468061150457, 0.2502087260548593, 0.2519723181662847, 0.2510646570984622, 0.24735875778771343, 0.24462183135786864, 0.24305323050653233, 0.24351380723329152, 0.24625706187735585, 0.24734398882255484, 0.24653545463680204, 0.24611626217900792, 0.24936908319716825, 0.24424766154699185, 0.2422968579972922, 0.24296794685528927, 0.23982078748470176, 0.23882868365221288, 0.24057649203336, 0.24338062669213797, 0.24159170745649505, 0.2452235599652401, 0.24435601356287046, 0.2494132344960503, 0.24803203419202555, 0.24772840954679837, 0.2456880559130185, 0.24486578401359976, 0.24372205790668483, 0.248238012106327, 0.2511196418801914, 0.2562253073632691, 0.25025442353904015, 0.24999537121075185, 0.2497164835108917, 0.2534648564076663, 0.25757465838935145, 0.2532048768956617, 0.24950225029818612, 0.24842513985424397, 0.24610928345014893, 0.24025482071897605, 0.23873455792779819, 0.24079073789818672, 0.24363810112866988]
x_PD115 = [640, 1280, 1920, 2560, 3200, 3840, 4480, 5120, 5760, 6400, 7040, 7680, 8320, 8960, 9600, 10240, 10880, 11520, 12160, 12800, 13440, 14080, 14720, 15360, 16000, 16640, 17280, 17920, 18560, 19200, 19840, 20480, 21120, 21760, 22400, 23040, 23680, 24320, 24960, 25600, 26240, 26880, 27520, 28160, 28800, 29440, 30080, 30720, 31360, 32000, 32640, 33280, 33920, 34560, 35200, 35840, 36480, 37120, 37760, 38400, 39040, 39680, 40320, 40960, 41600, 42240, 42880, 43520, 44160, 44800, 45440, 46080, 46720, 47360, 48000, 48640, 49280, 49920, 50560, 51200, 51840, 52480, 53120, 53760, 54400, 55040, 55680, 56320, 56960, 57600, 58240, 58880, 59520, 60160, 60800, 61440, 62080, 62720, 63360, 64000, 64640, 65280, 65920, 66560, 67200, 67840, 68480, 69120, 69760, 70400, 71040, 71680, 72320, 72960, 73600]
history_PD115 = [0.5350429, 0.51949847, 0.44707847, 0.3653275, 0.3511468, 0.36962867, 0.38111752, 0.41798535, 0.2989656, 0.26630113, 0.29905817, 0.31303126, 0.3231901, 0.3258544, 0.31800982, 0.28164893, 0.2914116, 0.28335533, 0.27528414, 0.3080245, 0.30675164, 0.28627792, 0.31965473, 0.2760962, 0.28484777, 0.2834057, 0.2714973, 0.26997834, 0.32857212, 0.2933649, 0.28763157, 0.2687203, 0.25654697, 0.31151387, 0.27287993, 0.31717527, 0.24233246, 0.29335806, 0.25901324, 0.23048906, 0.23293386, 0.24543433, 0.31600448, 0.30451724, 0.2300374, 0.26252946, 0.21126981, 0.20697065, 0.24276625, 0.22673821, 0.22153687, 0.2695132, 0.2715039, 0.21549188, 0.24058007, 0.27629653, 0.2957898, 0.25800624, 0.25747424, 0.21046816, 0.22600892, 0.24095826, 0.24197653, 0.19924489, 0.256502, 0.25758028, 0.25747654, 0.20943642, 0.27026233, 0.31126842, 0.28249648, 0.19939514, 0.26784465, 0.24289571, 0.21400566, 0.2199895, 0.22893582, 0.247659, 0.27094635, 0.25712633, 0.23925865, 0.24234353, 0.27864447, 0.19815487, 0.22473963, 0.24900775, 0.21149635, 0.22989975, 0.25630677, 0.26861784, 0.22549143, 0.27791023, 0.2365481, 0.29492822, 0.23560123, 0.24499579, 0.22732487, 0.23746534, 0.23342852, 0.2888816, 0.2770543, 0.3021763, 0.19651647, 0.2476639, 0.2472065, 0.2872002, 0.29456288, 0.21387684, 0.21617861, 0.23873115, 0.22526658, 0.18756466, 0.2250522, 0.25929636, 0.26926437]



## PD 115 without Reg
# seed 74
ema_PD115_nr = [0.53825915, 0.5382591485977173, 0.5335141867399216, 0.5244838669896126, 0.508424261957407, 0.49281837619841096, 0.4797417111173272, 0.4696703362740577, 0.4636418707618863, 0.447998117278494, 0.43114286440951605, 0.42039907972523255, 0.4115504443690957, 0.40273636162690357, 0.3958662502206448, 0.38804983069632204, 0.37944798144817726, 0.369781239090446, 0.3597665607256443, 0.3520985674252585, 0.3466159699610301, 0.340463805042335, 0.33351753161188535, 0.3312950760647944, 0.3245880660884343, 0.31899329639727736, 0.31286365345646316, 0.3071719427901152, 0.3003992769120017, 0.3014283656774055, 0.30230561313314064, 0.30091641918375633, 0.29740216054518115, 0.29292793915866994, 0.2896727458792852, 0.285264773919397, 0.287017247457099, 0.28106967781078107, 0.28222847140513024, 0.27976502618207477, 0.2769962338299806, 0.2739082459531357, 0.26879439359207125, 0.27113963608711217, 0.2722828606793516, 0.27045282602571546, 0.2703919698023477, 0.2673431133529853, 0.26237769829940594, 0.26211472369174776, 0.25968666389829476, 0.2591705636553159, 0.2611346110438279, 0.2603800265254055, 0.2585913436050541, 0.2574413937123412, 0.2590286125354583, 0.26347287502401606, 0.265622621832909, 0.2646594220923977, 0.2593733228320715, 0.25601721320225335, 0.2579109327227491, 0.25813859011804624, 0.25268320118852466, 0.2565968502354742, 0.2579259576313604, 0.2577415646832802, 0.2534503765453355, 0.2556136751964202, 0.26259872360180414, 0.26430178621247946, 0.25914152507932964, 0.2627502505029915, 0.2617949209254097, 0.25616590633759895, 0.254591700030297, 0.2534724249776572, 0.2519329375925395, 0.25412112059319614, 0.2546068601202184, 0.2543689256064189, 0.25654250191021793, 0.26167801445456296, 0.2590089612519617, 0.2582155412419923, 0.26132174247362805, 0.2556130833672755, 0.25582638209288777, 0.2569670393338096, 0.2579240990522765, 0.25445823307424154, 0.2607423642479423, 0.26317994706657916, 0.2678893617942398, 0.26643724550197706, 0.26818498351624775, 0.2681307617883489, 0.2660552538091486, 0.26531653195511207, 0.2695292018940232, 0.27127007392855124, 0.2747816987140027, 0.2692325217100722, 0.26915851556415044, 0.268311285881591, 0.27148792497073904, 0.27370123879690733, 0.26591533407491236, 0.2644295720898073, 0.26360142189052355, 0.2618113782207964, 0.25700060661442264, 0.2566419047338413, 0.2579934062560779, 0.2611111338734969]
x_PD115_nr = [640, 1280, 1920, 2560, 3200, 3840, 4480, 5120, 5760, 6400, 7040, 7680, 8320, 8960, 9600, 10240, 10880, 11520, 12160, 12800, 13440, 14080, 14720, 15360, 16000, 16640, 17280, 17920, 18560, 19200, 19840, 20480, 21120, 21760, 22400, 23040, 23680, 24320, 24960, 25600, 26240, 26880, 27520, 28160, 28800, 29440, 30080, 30720, 31360, 32000, 32640, 33280, 33920, 34560, 35200, 35840, 36480, 37120, 37760, 38400, 39040, 39680, 40320, 40960, 41600, 42240, 42880, 43520, 44160, 44800, 45440, 46080, 46720, 47360, 48000, 48640, 49280, 49920, 50560, 51200, 51840, 52480, 53120, 53760, 54400, 55040, 55680, 56320, 56960, 57600, 58240, 58880, 59520, 60160, 60800, 61440, 62080, 62720, 63360, 64000, 64640, 65280, 65920, 66560, 67200, 67840, 68480, 69120, 69760, 70400, 71040, 71680, 72320, 72960, 73600]
history_PD115_nr = [0.53825915, 0.49080953, 0.443211, 0.36388782, 0.3523654, 0.36205173, 0.37902796, 0.40938568, 0.30720434, 0.2794456, 0.32370502, 0.33191273, 0.32340962, 0.33403525, 0.31770205, 0.30203134, 0.28278056, 0.26963446, 0.28308663, 0.2972726, 0.28509432, 0.27100107, 0.31129298, 0.26422498, 0.26864037, 0.25769687, 0.25594655, 0.23944528, 0.31069016, 0.31020084, 0.28841367, 0.26577383, 0.25265995, 0.260376, 0.24559303, 0.3027895, 0.22754155, 0.2926576, 0.25759402, 0.2520771, 0.24611636, 0.22276972, 0.29224682, 0.28257188, 0.2539825, 0.26984426, 0.2399034, 0.21768896, 0.25974795, 0.23783413, 0.25452566, 0.27881104, 0.25358877, 0.2424932, 0.24709184, 0.27331358, 0.30347124, 0.28497034, 0.25599062, 0.21179843, 0.22581223, 0.2749544, 0.2601875, 0.2035847, 0.2918197, 0.26988792, 0.25608203, 0.21482968, 0.27508336, 0.32546416, 0.27962935, 0.21269917, 0.29522878, 0.25319695, 0.20550478, 0.24042384, 0.24339895, 0.23807755, 0.27381477, 0.25897852, 0.25222751, 0.2761047, 0.30789763, 0.23498748, 0.25107476, 0.28927755, 0.20423515, 0.25774607, 0.26723295, 0.26653764, 0.22326544, 0.31729954, 0.2851182, 0.3102741, 0.2533682, 0.28391463, 0.26764277, 0.24737568, 0.25866804, 0.30744323, 0.28693792, 0.30638632, 0.21928993, 0.26849246, 0.26068622, 0.30007768, 0.29362106, 0.19584219, 0.2510577, 0.25614807, 0.24570099, 0.21370366, 0.2534136, 0.27015692, 0.28917068]


## Grid Search with Reg
ema_GS_reg = [0.5646509, 0.5646508932113647, 0.5579137057065964, 0.5419458642601966, 0.5295922215282917, 0.514656521871686, 0.5008631032660603, 0.4896589739878475, 0.4752416780972033, 0.460153913919647, 0.44705350377367686, 0.43913057307785824, 0.4348314123784251, 0.430309487947948, 0.42392752489667584, 0.4132362477888595, 0.41101954933195683, 0.4025619347946886, 0.3965393948605994, 0.39183915985089596, 0.38538722693141153, 0.3871319070388304, 0.37719305827582716, 0.3740695489141182, 0.3748653059625181, 0.37619058717103776, 0.37536566938887816, 0.36802879708347425, 0.36399144891504387, 0.3619825777137632, 0.35801140398302866, 0.3579028898225951, 0.35734103504539083, 0.3543623034690668, 0.3595311413289472, 0.3608973446561126, 0.3600635012751724, 0.36042285091572157, 0.3583863359113916, 0.355823577769433, 0.35119965195164193, 0.35097636448894537, 0.34770003557999074, 0.3436643266749305, 0.33996383703398014, 0.33454473933560225, 0.3332426856844059, 0.33084995247916266, 0.3332663479268183, 0.33418374980863597]
x_GS_reg = [3200, 6400, 9600, 12800, 16000, 19200, 22400, 25600, 28800, 32000, 35200, 38400, 41600, 44800, 48000, 51200, 54400, 57600, 60800, 64000, 67200, 70400, 73600, 76800, 80000, 83200, 86400, 89600, 92800, 96000, 99200, 102400, 105600, 108800, 112000, 115200, 118400, 121600, 124800, 128000, 131200, 134400, 137600, 140800, 144000, 147200, 150400, 153600, 156800]
history_GS_reg = [0.5646509, 0.49727902, 0.3982353, 0.41840944, 0.38023522, 0.37672234, 0.3888218, 0.34548602, 0.32436404, 0.3291498, 0.3678242, 0.39613897, 0.38961217, 0.36648986, 0.31701475, 0.39106926, 0.3264434, 0.34233654, 0.34953704, 0.32731983, 0.40283403, 0.28774342, 0.34595796, 0.38202712, 0.38811812, 0.3679414, 0.30199695, 0.32765532, 0.34390274, 0.32227084, 0.35692626, 0.35228434, 0.32755372, 0.40605068, 0.37319317, 0.3525589, 0.363657, 0.3400577, 0.33275875, 0.30958432, 0.34896678, 0.31821308, 0.30734295, 0.30665943, 0.28577286, 0.3215242, 0.30931535, 0.3550139, 0.34244037]

ema_GS = [0.60191965, 0.6019196510314941, 0.5915467321872712, 0.5712674301862717, 0.5589803047776222, 0.5436787326157093, 0.5288852857667208, 0.5174447304413915, 0.5004676675563634, 0.48620679042627635, 0.47264857347509465, 0.4642135404560077, 0.4610054585051335, 0.4530209832827864, 0.4479056060410098, 0.4366634631243249, 0.4339177051860303, 0.4281948482702562, 0.42155918559822936, 0.41697893215471654, 0.411047941505201, 0.4157004738384074, 0.408908163118385, 0.4075549941730703, 0.40713736428071906, 0.40840760481282934, 0.41258012007407724, 0.4086886800891076, 0.40386966981583067, 0.40284506784164814, 0.397644525524532, 0.3997034839599328, 0.3979798208304343, 0.39570748066557326, 0.3979234092442949, 0.39860503909070133, 0.399350673291792, 0.4020585063103225, 0.3990703144374537, 0.40103644299832414, 0.39665767700964893, 0.39764911667848846, 0.39825933339904446, 0.393530674243367, 0.3981739319542613, 0.3947912517648548, 0.3966713097806224, 0.39366299037177405, 0.3988130699757252, 0.4022170464244646]
x_GS = [3200, 6400, 9600, 12800, 16000, 19200, 22400, 25600, 28800, 32000, 35200, 38400, 41600, 44800, 48000, 51200, 54400, 57600, 60800, 64000, 67200, 70400, 73600, 76800, 80000, 83200, 86400, 89600, 92800, 96000, 99200, 102400, 105600, 108800, 112000, 115200, 118400, 121600, 124800, 128000, 131200, 134400, 137600, 140800, 144000, 147200, 150400, 153600, 156800]
x_GS = [i / 5 for i in x_GS]
history_GS = [0.60191965, 0.49819046, 0.3887537, 0.44839618, 0.40596458, 0.39574426, 0.41447973, 0.3476741, 0.3578589, 0.35062462, 0.38829824, 0.43213272, 0.3811607, 0.4018672, 0.33548418, 0.40920588, 0.37668914, 0.36183822, 0.37575665, 0.35766903, 0.45757326, 0.34777737, 0.39537647, 0.4033787, 0.41983977, 0.45013276, 0.37366572, 0.36049858, 0.39362365, 0.35083964, 0.4182341, 0.38246685, 0.37525642, 0.41786677, 0.4047397, 0.40606138, 0.426429, 0.3721766, 0.4187316, 0.35724878, 0.40657207, 0.40375128, 0.35097274, 0.43996325, 0.36434713, 0.41359183, 0.36658812, 0.4451638, 0.43285283]



ema_KT = [0.75, 0.75, 0.3428066670894623, 0.3308306336402893, 0.5133]
x_KT = [0, 47736, 47736, 48672, 48672 + 936]

# history_KT = 




fig, ax = plt.subplots()


ax.plot(x_ESGD15, ema_ESGD15[:len(history_ESGD15)], ls=':', label = "ESGD")
ax.plot(x_PD115_nr, ema_PD115_nr[:len(history_PD115_nr)], label = "Population Descent")
# ax.plot(x_PD, ema_PD[:len(history_PD)], label = "Population Descent")
ax.plot(x_GS, ema_GS[:len(history_GS)], ls='--', label = "Grid Search")
ax.plot(x_KT, ema_KT, ls='-.', label = "KT RandomSearch")

# labels
plt.title("FMNIST Val Loss Progress")
plt.xlabel("Gradient Steps")
plt.ylabel("Validation Loss")
plt.tight_layout()

# legend
leg = ax.legend();

plt.show()





def graph_history(history):
	integers = [i for i in range(1, (len(history))+1)]

	ema = []
	avg = history[0]

	ema.append(avg)

	for loss in history:
		avg = (avg * 0.9) + (0.1 * loss)
		ema.append(avg)


	x = [j * 938 for j in integers]
	y = history

	# plot line
	plt.plot(x, ema[:len(history)])
	# plot title/captions
	plt.title("FMNIST Val Loss")
	plt.xlabel("Gradient Steps")
	plt.ylabel("Validation Loss")
	plt.tight_layout()


	print("ema:"), print(ema), print("")
	print("x:"), print(x), print("")
	print("history:"), print(history), print("")


	
	# plt.savefig("TEST_DATA/PD_trial_%s.png" % trial)
	def save_image(filename):
	    p = PdfPages(filename)
	    fig = plt.figure(1)
	    fig.savefig(p, format='pdf') 
	    p.close()

	filename = "FMNIST_progress_with_reg_model4_line.pdf"
	save_image(filename)

	# plot points too
	plt.scatter(x, history, s=20)

	def save_image(filename):
	    p = PdfPages(filename)
	    fig = plt.figure(1)
	    fig.savefig(p, format='pdf') 
	    p.close()

	filename = "FMNIST_progress_with_reg_model4_with_points.pdf"
	save_image(filename)


	plt.show(block=True), plt.close()
	plt.close('all')



