# Inverse Graph Convolution Networks

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin
# Results for 10 runs
# test_acc           = 0.8431999146938324 +- 0.006823517696556644
# test_cross_entropy = 0.8292202174663543 +- 0.008054932442014235
# test_loss          = 1.5021563529968263 +- 0.005806926082195405
# val_acc            = 0.8127999007701874 +- 0.00552811730609111
# val_cross_entropy  = 0.85521280169487 +- 0.007545950245539115
# val_loss           = 1.5281489610671997 +- 0.006822854155478488
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/lazy.gin
# Results for 10 runs
# test_acc           = 0.8428999066352845 +- 0.006394531873513696
# test_cross_entropy = 0.8318553864955902 +- 0.01124307259597092
# test_loss          = 1.5033095955848694 +- 0.006010301151108756
# val_acc            = 0.8127999007701874 +- 0.005528119462544703
# val_cross_entropy  = 0.8583055973052979 +- 0.010101142499951626
# val_loss           = 1.5297598242759705 +- 0.005903132518118667
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/power-series.gin --bindings="num_terms=10"
# Results for 10 runs
# test_acc           = 0.841399896144867 +- 0.00635923647426896
# test_cross_entropy = 0.990231728553772 +- 0.008562806679297985
# test_loss          = 1.6748472809791566 +- 0.008500875306277642
# val_acc            = 0.8131999015808106 +- 0.006209662881649818
# val_cross_entropy  = 1.012561595439911 +- 0.006778961449834772
# val_loss           = 1.6971771597862244 +- 0.006822298369122665
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/low-rank.gin --bindings='rank=100'
# Results for 10 runs
# test_acc           = 0.7615998983383179 +- 0.00869710801842641
# test_cross_entropy = 0.9554973602294922 +- 0.008260781250730943
# test_loss          = 1.5954362869262695 +- 0.013915188961715431
# val_acc            = 0.745999938249588 +- 0.007483301265030819
# val_cross_entropy  = 1.0238064408302308 +- 0.0053843948626046394
# val_loss           = 1.663745355606079 +- 0.01238682217299224
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/low-rank.gin --bindings='rank=200'
# Results for 10 runs
# test_acc           = 0.8147999286651612 +- 0.005015972407466588
# test_cross_entropy = 0.8575874030590057 +- 0.01001719904523011
# test_loss          = 1.5095795392990112 +- 0.01028357074824637
# val_acc            = 0.7637998938560486 +- 0.007820469375296603
# val_cross_entropy  = 0.9493454098701477 +- 0.005956680142266275
# val_loss           = 1.6013375401496888 +- 0.0085913916582077
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/low-rank.gin --bindings='rank=1000'
# Results for 10 runs
# test_acc           = 0.842299884557724 +- 0.004648662544057939
# test_cross_entropy = 0.8391031265258789 +- 0.007934331230924647
# test_loss          = 1.514020073413849 +- 0.00329140703643032
# val_acc            = 0.8137999057769776 +- 0.0066000002803531955
# val_cross_entropy  = 0.8603773891925812 +- 0.007324121572929028
# val_loss           = 1.5352943539619446 +- 0.0030948183408107213
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/ogbn-arxiv/lazy.gin
# Results for 10 runs
# test_acc           = 0.711962240934372 +- 0.0013800993204688878
# test_cross_entropy = 0.9537405431270599 +- 0.0027428242634262657
# test_loss          = 0.9537405431270599 +- 0.0027428242634262657
# val_acc            = 0.7213463723659516 +- 0.0020571100337344727
# val_cross_entropy  = 0.9285373151302337 +- 0.0040098876701630546
# val_loss           = 0.928537255525589 +- 0.00400987447228198
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/ogbn-arxiv/lazy.gin --bindings='epsilon=0.05'
# Results for 10 runs
# test_acc           = 0.7114684462547303 +- 0.0017040454338729193
# test_cross_entropy = 0.971020781993866 +- 0.003961086869293576
# test_loss          = 0.9710207760334015 +- 0.0039610888802275165
# val_acc            = 0.7206215262413025 +- 0.0009447424945354277
# val_cross_entropy  = 0.949244749546051 +- 0.003543130164321587
# val_loss           = 0.9492446899414062 +- 0.003543130164321587
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/ogbn-arxiv/lazy.gin --bindings='epsilon=0.2'
# Results for 10 runs
# test_acc           = 0.6993786454200744 +- 0.002346696752627672
# test_cross_entropy = 0.9809624969959259 +- 0.003964673675094083
# test_loss          = 0.980962485074997 +- 0.003964670072281538
# val_acc            = 0.7113057553768158 +- 0.0016876702252022727
# val_cross_entropy  = 0.9472987592220307 +- 0.002420426133449007
# val_loss           = 0.9472986996173859 +- 0.002420426133449007
```

### Quadratic Loss Approximation

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/losses/quad.gin
# Results for 10 runs
# test_acc                = 0.8411999106407165 +- 0.005653338239649189
# test_cross_entropy      = 0.8033395230770111 +- 0.0125039367969246
# test_loss               = -0.5175256788730621 +- 0.010373902016212204
# test_quad_cross_entropy = -1.3078410625457764 +- 0.015884317861397556
# val_acc                 = 0.8147998929023743 +- 0.006997122380366498
# val_cross_entropy       = 0.8313239455223084 +- 0.009865571037673757
# val_loss                = -0.482570606470108 +- 0.0067368741425244
# val_quad_cross_entropy  = -1.2728859901428222 +- 0.012451119134661565
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/losses/quad.gin --bindings="monitor='val_quad_cross_entropy'"
# Results for 10 runs
# test_acc                = 0.8425999224185944 +- 0.005161402386702602
# test_cross_entropy      = 0.8018308579921722 +- 0.012796623277143952
# test_loss               = -0.5191118478775024 +- 0.010720701674378767
# test_quad_cross_entropy = -1.3114154100418092 +- 0.0152175472495464
# val_acc                 = 0.8163998782634735 +- 0.007031354949915743
# val_cross_entropy       = 0.8306720018386841 +- 0.010242598069772867
# val_loss                = -0.48284260630607606 +- 0.007584924795358441
# val_quad_cross_entropy  = -1.2751461744308472 +- 0.011897064441671759
python -m graph_tf gtf_config/build_and_fit.gin igcn/config/ogbn-arxiv/lazy.gin igcn/config/losses/quad.gin
# Final results
# test_acc                : 0.6835587024688721
# test_cross_entropy      : 2.0225040912628174
# test_loss               : -7.928796291351318
# test_quad_cross_entropy : -7.928796291351318
# val_acc                 : 0.6983792185783386
# val_cross_entropy       : 2.0497653484344482
# val_loss                : -8.281379699707031
# val_quad_cross_entropy  : -8.281380653381348
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/ogbn-arxiv/lazy.gin igcn/config/losses/quad.gin --bindings="monitor='val_quad_cross_entropy'"
# Results for 10 runs
# test_acc                = 0.6830319404602051 +- 0.0030406504323333577
# test_cross_entropy      = 2.1924077749252318 +- 0.027888281949796652
# test_loss               = -8.362832736968993 +- 0.03550705779847601
# test_quad_cross_entropy = -8.362832736968993 +- 0.03550705779847601
# val_acc                 = 0.7019732534885407 +- 0.0023339049922464776
# val_cross_entropy       = 2.17172110080719 +- 0.01753800839393432
# val_loss                = -8.754856586456299 +- 0.0419640460365712
# val_quad_cross_entropy  = -8.754857540130615 +- 0.0419640460365712
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/ogbn-arxiv/lazy.gin igcn/config/losses/quad.gin --bindings="monitor='val_acc'"
# Results for 10 runs
# test_acc                = 0.6930518805980682 +- 0.002325356785570999
# test_cross_entropy      = 2.070184755325317 +- 0.016929145713854375
# test_loss               = -8.214222717285157 +- 0.07198958273895718
# test_quad_cross_entropy = -8.214222717285157 +- 0.07198958273895718
# val_acc                 = 0.7073425650596619 +- 0.0023120730641897898
# val_cross_entropy       = 2.0980108261108397 +- 0.018247161416253013
# val_loss                = -8.573719120025634 +- 0.07714926061671426
# val_quad_cross_entropy  = -8.57372007369995 +- 0.07714926061671426
```

### Subgraph Batching

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-lp.gin --bindings='batch_size=512'
# Results for 10 runs
# test_acc           = 0.800999891757965 +- 0.021052333252612094
# test_cross_entropy = 1.117299699783325 +- 0.018184258536373663
# test_loss          = 1.602143406867981 +- 0.009867517319037832
# val_acc            = 0.7741998910903931 +- 0.013840506195292234
# val_cross_entropy  = 1.1297026753425599 +- 0.016477431154419086
# val_loss           = 1.6145463585853577 +- 0.0086587490060483
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-lp.gin --bindings='batch_size=1024'
# Results for 10 runs
# test_acc           = 0.8264999032020569 +- 0.011289362446914471
# test_cross_entropy = 0.9257604956626893 +- 0.011026718164481754
# test_loss          = 1.5359347105026244 +- 0.011546322454056578
# val_acc            = 0.7989998996257782 +- 0.006767576835652836
# val_cross_entropy  = 0.9427375793457031 +- 0.008727889174236137
# val_loss           = 1.5529118418693542 +- 0.010957806200425349
```

#### V3

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-v3-lp.gin --bindings='
    batch_size=512
    temperature=1.0
'
# Results for 10 runs
# test_acc           = 0.7316999137401581 +- 0.04705751934866201
# test_cross_entropy = 1.330195999145508 +- 0.020673895303470823
# test_loss          = 1.6701044201850892 +- 0.015714610291383962
# val_acc            = 0.703999936580658 +- 0.053985167131604596
# val_cross_entropy  = 1.3430036783218384 +- 0.02175332114532694
# val_loss           = 1.6829120874404908 +- 0.01615947087436165
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-v3-lp.gin --bindings='
    batch_size=512
    temperature=0.1
'
# Results for 10 runs
# test_acc           = 0.8247998714447021 +- 0.016148079944057495
# test_cross_entropy = 0.9646491408348083 +- 0.009152529111504637
# test_loss          = 1.5799957752227782 +- 0.005548840845772949
# val_acc            = 0.7965998947620392 +- 0.012619020132143166
# val_cross_entropy  = 0.9877132296562194 +- 0.008391601805716017
# val_loss           = 1.6030598640441895 +- 0.004147571567606301
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-v3-lp.gin --bindings='
    batch_size=512
    temperature=0.01
'
# Results for 10 runs
# test_acc           = 0.8251998901367188 +- 0.0120648168973984
# test_cross_entropy = 0.9674132764339447 +- 0.010250000751890526
# test_loss          = 1.5862084746360778 +- 0.008070560953019945
# val_acc            = 0.7941999197006225 +- 0.012408073926980064
# val_cross_entropy  = 0.9890770375728607 +- 0.011200885028500149
# val_loss           = 1.6078722476959229 +- 0.009124262511076689
## With quadratic loss
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-v3-lp.gin igcn/config/losses/quad.gin  --bindings='
    batch_size=512
    temperature=0.01
'
# Results for 10 runs
# test_acc           = 0.8249999105930328 +- 0.018000006675740938
# test_cross_entropy = 0.9384763658046722 +- 0.012741190460553052
# test_loss          = 1.5547003388404845 +- 0.011954570752346258
# val_acc            = 0.7963998854160309 +- 0.015278762663659603
# val_cross_entropy  = 0.9714768648147583 +- 0.012418086533772325
# val_loss           = 1.5970982909202576 +- 0.013010326471046784
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-v3-lp.gin igcn/config/losses/quad.gin  --bindings='
    batch_size=512
    temperature=0.001
'
# Results for 10 runs
# test_acc           = 0.8213998913764954 +- 0.00873154783187623
# test_cross_entropy = 0.9307237565517426 +- 0.0061118722666516434
# test_loss          = 1.5560230016708374 +- 0.010349119870909546
# val_acc            = 0.797999906539917 +- 0.014532713418121092
# val_cross_entropy  = 0.9586847424507141 +- 0.006555373701446323
# val_loss           = 1.5930785894393922 +- 0.012491982554511812
```

Experimenting with batch size:

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-v3-lp.gin igcn/config/losses/quad.gin  --bindings='
    batch_size=128
    temperature=0.01
'
# Results for 10 runs
# test_acc           = 0.506099933385849 +- 0.057821197855218755
# test_cross_entropy = 1.621759843826294 +- 0.02390285935644103
# test_loss          = 1.9177039742469788 +- 0.01800597569335669
# val_acc            = 0.48279992341995237 +- 0.045958242789305055
# val_cross_entropy  = 1.646865177154541 +- 0.020435948533054672
# val_loss           = 1.9462181091308595 +- 0.01632198048335318
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-v3-lp.gin igcn/config/losses/quad.gin  --bindings='
    batch_size=256
    temperature=0.01
'
# Results for 10 runs
# test_acc           = 0.7500999331474304 +- 0.03780593438593974
# test_cross_entropy = 1.2305927991867065 +- 0.02538894171931058
# test_loss          = 1.722992193698883 +- 0.014827709457793242
# val_acc            = 0.7287999451160431 +- 0.027541968303047226
# val_cross_entropy  = 1.252080774307251 +- 0.023833084720241392
# val_loss           = 1.746661329269409 +- 0.013637481123049031
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-v3-lp.gin igcn/config/losses/quad.gin  --bindings='
    batch_size=512
    temperature=0.01
'
# Results for 10 runs
# test_acc           = 0.8249999105930328 +- 0.018000006675740938
# test_cross_entropy = 0.9384763658046722 +- 0.012741190460553052
# test_loss          = 1.5547003388404845 +- 0.011954570752346258
# val_acc            = 0.7963998854160309 +- 0.015278762663659603
# val_cross_entropy  = 0.9714768648147583 +- 0.012418086533772325
# val_loss           = 1.5970982909202576 +- 0.013010326471046784
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-v3-lp.gin igcn/config/losses/quad.gin  --bindings='
    batch_size=1024
    temperature=0.01
'
# Results for 10 runs
# test_acc           = 0.8381999135017395 +- 0.007152626309012966
# test_cross_entropy = 0.8298879981040954 +- 0.011990418951335879
# test_loss          = 1.48286052942276 +- 0.009288953502227488
# val_acc            = 0.8185999035835266 +- 0.011766054773696424
# val_cross_entropy  = 0.8595864295959472 +- 0.012352321034282908
# val_loss           = 1.5222928047180175 +- 0.009225708982764695
```

### Combined Quadratic Loss / Subgraph Batching

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/batched-lp.gin igcn/config/losses/quad.gin --bindings='batch_size=512'
# Results for 10 runs
# test_acc           = 0.8036998987197876 +- 0.017759802619035866
# test_cross_entropy = 1.0845224261283875 +- 0.028651732628534606
# test_loss          = 1.5868109583854675 +- 0.015964339169551395
# val_acc            = 0.7789998769760131 +- 0.011636149304132865
# val_cross_entropy  = 1.0989405512809753 +- 0.025654915902101352
# val_loss           = 1.6039423584938048 +- 0.012650519713555571
```
