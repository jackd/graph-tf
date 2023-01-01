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
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/ogbn-arxiv/lazy.gin \
--bindings='
    dropout_rate = 0.5
    hidden_units = [1024, 512]
    input_dropout_rate = 0
    activation = @gtf.utils.models.prelu
    lr = 2e-3
'
# Results for 10 runs
# test_acc           = 0.7161286473274231 +- 0.0025397897810463554
# test_cross_entropy = 0.9138263821601867 +- 0.00496579910710566
# test_loss          = 0.9138263881206512 +- 0.004965807240956798
# val_acc            = 0.7303064286708831 +- 0.0012159245317773944
# val_cross_entropy  = 0.8782849431037902 +- 0.0025077329234558044
# val_loss           = 0.8782848834991455 +- 0.0025077232637869553

python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/ogbn-arxiv/lazy-v2.gin
# Completed 10 trials
# test_acc           : 0.7133860170841217 +- 0.003125309704871446
# test_cross_entropy : 0.9414996862411499 +- 0.007347032192700789
# test_loss          : 0.9414996981620789 +- 0.007347026555702601
# val_acc            : 0.7240142524242401 +- 0.0009663739832727263
# val_cross_entropy  : 0.90623459815979 +- 0.0032755681053393965
# val_loss           : 0.9062345385551452 +- 0.0032755681053393965
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/ogbn-arxiv/lazy-v2.gin --bindings='tol=1e-3'
# Completed 10 trials
# test_acc           : 0.7149599850177765 +- 0.0027676591869668066
# test_cross_entropy : 0.9361138880252838 +- 0.006171773586868322
# test_loss          : 0.9361138999462127 +- 0.006171786520139506
# val_acc            : 0.7241484820842743 +- 0.0009304116361227964
# val_cross_entropy  : 0.9030679941177369 +- 0.003173223865107103
# val_loss           : 0.9030679345130921 +- 0.003173223865107103
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/ogbn-arxiv/lazy.gin \
--bindings='
    dropout_rate = 0.5
    hidden_units = [1024, 512]
    input_dropout_rate = 0
    activation = @gtf.utils.models.prelu
    lr = 2e-3
    mlp.normalization=None
    renormalized=False
'
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

## DAGNN PPR

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin
# Completed 10 trials
# test_acc           : 0.8470999121665954 +- 0.0031448524622474903
# test_cross_entropy : 0.5986384510993957 +- 0.005906312101587842
# test_loss          : 1.1763751268386842 +- 0.004957391095713499
# val_acc            : 0.8225999236106872 +- 0.006199990934007383
# val_cross_entropy  : 0.6401195108890534 +- 0.004342835830853232
# val_loss           : 1.217856240272522 +- 0.006557012535154112
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin
# Completed 10 trials
# test_acc           : 0.7348999738693237 +- 0.007542551851032817
# test_cross_entropy : 1.1621787190437316 +- 0.0052120214553928144
# test_loss          : 1.8362551212310791 +- 0.005027293930874139
# val_acc            : 0.7347999334335327 +- 0.010087596907114268
# val_cross_entropy  : 1.1808425784111023 +- 0.005788040954767149
# val_loss           : 1.8549189805984496 +- 0.005343847568906561
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin
# Completed 10 trials
# test_acc           : 0.8046999096870422 +- 0.0017349090417887252
# test_cross_entropy : 0.5168327510356903 +- 0.006481546934862003
# test_loss          : 0.7212450325489044 +- 0.011941931268452877
# val_acc            : 0.8213999032974243 +- 0.008534646118343833
# val_cross_entropy  : 0.4817315131425858 +- 0.003190635504505322
# val_loss           : 0.6861437857151031 +- 0.0075135754977480665
```

## DAGNN Pinv

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/pinv.gin --bindings='l2_reg=1e-5'
# Results for 10 runs
# test_acc           = 0.8108998775482178 +- 0.005243104017101028
# test_cross_entropy = 0.6086649179458619 +- 0.012957802192529096
# test_loss          = 0.7771891057491302 +- 0.007078932491708424
# val_acc            = 0.8059998989105225 +- 0.00753659432766515
# val_cross_entropy  = 0.6869994044303894 +- 0.011294513801871898
# val_loss           = 0.8555235862731934 +- 0.00965409197058636
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin igcn/config/impl/pinv.gin --bindings='l2_reg=1e-5'
# Results for 10 runs
# test_acc           = 0.6000999450683594 +- 0.01768304066498091
# test_cross_entropy = 1.2143028616905212 +- 0.027535249192441234
# test_loss          = 1.3109641790390014 +- 0.026788703089953254
# val_acc            = 0.6181999206542969 +- 0.010017974798799515
# val_cross_entropy  = 1.2157050609588622 +- 0.026045322675680164
# val_loss           = 1.3123663902282714 +- 0.025363004806477785
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/impl/pinv.gin --bindings='l2_reg=1e-5'
# Results for 10 runs
# test_acc           = 0.7974998891353607 +- 0.003640053929532806
# test_cross_entropy = 0.5988308846950531 +- 0.004151997150887548
# test_loss          = 0.6492764353752136 +- 0.0061281003413587755
# val_acc            = 0.820399922132492 +- 0.002800003120041189
# val_cross_entropy  = 0.5249153017997742 +- 0.003454977960647511
# val_loss           = 0.5753608584403992 +- 0.0035889443938851326
```

### DAGNN Heat

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/heat.gin --bindings='
l2_reg=2.5e-4
t=3.05
'
# Results for 10 runs
# test_acc           = 0.8395999014377594 +- 0.003039738481164253
# test_cross_entropy = 0.6502970933914185 +- 0.01507294579728569
# test_loss          = 1.0849357843399048 +- 0.007335427352401799
# val_acc            = 0.8123999059200286 +- 0.005851491910860191
# val_cross_entropy  = 0.6940236568450928 +- 0.013475748922218785
# val_loss           = 1.1286623358726502 +- 0.007331685944227427
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin igcn/config/impl/heat.gin --bindings='
l2_reg=1e-3
t=3.05
'
# Results for 10 runs
# test_acc           = 0.715699952840805 +- 0.0040012629413430836
# test_cross_entropy = 1.204524612426758 +- 0.0036018518155059207
# test_loss          = 1.8131738901138306 +- 0.004711048615818238
# val_acc            = 0.7281999588012695 +- 0.009357340924599498
# val_cross_entropy  = 1.2280866742134093 +- 0.0029240417712786664
# val_loss           = 1.8367359399795533 +- 0.004752867394586437
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/impl/heat.gin --bindings='
l2_reg=2.5e-4
t=3.05
'
# Results for 10 runs
# test_acc           = 0.7981998741626739 +- 0.0032802387301869617
# test_cross_entropy = 0.559614509344101 +- 0.00397595837281702
# test_loss          = 0.7744017779827118 +- 0.008635809990591674
# val_acc            = 0.8011998951435089 +- 0.00858833859727954
# val_cross_entropy  = 0.5308033764362335 +- 0.005351223600797911
# val_loss           = 0.7455906629562378 +- 0.005623062801218087
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/impl/heat.gin --bindings='
l2_reg=1.25e-4
t=5.186
'
# Results for 10 runs
# test_acc           : 0.7946998775005341 +- 0.002758623628025645
# test_cross_entropy : 0.5676609396934509 +- 0.004594675942046989
# test_loss          : 0.7208189785480499 +- 0.005116699164739251
# val_acc            : 0.8061998963356019 +- 0.004512212567855101
# val_cross_entropy  : 0.5293090045452118 +- 0.004292945523219227
# val_loss           : 0.6824670374393463 +- 0.0037035534176918764
```

#### DAGNN Renormalized Heat

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/heat.gin --bindings='
l2_reg=2.5e-4
t=3.05
renormalized=True
'
# Results for 10 runs
# test_acc           = 0.8395999014377594 +- 0.003039738481164253
# test_cross_entropy = 0.6502970933914185 +- 0.01507294579728569
# test_loss          = 1.0849357843399048 +- 0.007335427352401799
# val_acc            = 0.8123999059200286 +- 0.005851491910860191
# val_cross_entropy  = 0.6940236568450928 +- 0.013475748922218785
# val_loss           = 1.1286623358726502 +- 0.007331685944227427
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin igcn/config/impl/heat.gin --bindings='
l2_reg=1e-3
t=3.05
renormalized=True
'
# Results for 10 runs
# test_acc           = 0.715699952840805 +- 0.0040012629413430836
# test_cross_entropy = 1.204524612426758 +- 0.0036018518155059207
# test_loss          = 1.8131738901138306 +- 0.004711048615818238
# val_acc            = 0.7281999588012695 +- 0.009357340924599498
# val_cross_entropy  = 1.2280866742134093 +- 0.0029240417712786664
# val_loss           = 1.8367359399795533 +- 0.004752867394586437
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/impl/heat.gin --bindings='
l2_reg=2.5e-4
t=3.05
renormalized=True
'
# Results for 10 runs
# test_acc           = 0.7981998741626739 +- 0.0032802387301869617
# test_cross_entropy = 0.559614509344101 +- 0.00397595837281702
# test_loss          = 0.7744017779827118 +- 0.008635809990591674
# val_acc            = 0.8011998951435089 +- 0.00858833859727954
# val_cross_entropy  = 0.5308033764362335 +- 0.005351223600797911
# val_loss           = 0.7455906629562378 +- 0.005623062801218087
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/impl/heat.gin --bindings='
l2_reg=1.25e-4
t=5.186
renormalized=True
'
# Results for 10 runs
# test_acc           : 0.7946998775005341 +- 0.002758623628025645
# test_cross_entropy : 0.5676609396934509 +- 0.004594675942046989
# test_loss          : 0.7208189785480499 +- 0.005116699164739251
# val_acc            : 0.8061998963356019 +- 0.004512212567855101
# val_cross_entropy  : 0.5293090045452118 +- 0.004292945523219227
# val_loss           : 0.6824670374393463 +- 0.0037035534176918764
```

## GCNII PPR

`dropout_rate` increased by 0.1 to account for no intra-propagation dropout.

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/v2.gin --bindings='
    renormalized=True
    l2_reg = 2.5e-4
    dropout_rate=0.7
    rescaled=True
    test_preprocess=True
'
# Completed 10 trials
# test_acc           : 0.8504998862743378 +- 0.002906904257827969
# test_cross_entropy : 0.7709044575691223 +- 0.01498304654638068
# test_loss          : 1.1692887544631958 +- 0.008379913366222177
# val_acc            : 0.8277998924255371 +- 0.006095888268607835
# val_cross_entropy  : 0.8014185011386872 +- 0.015097594961642051
# val_loss           : 1.1998027920722962 +- 0.00862453892804352
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin igcn/config/impl/v2.gin --bindings='
    renormalized=True
    l2_reg = 2.5e-4
    dropout_rate=0.8
    rescaled=True
    hidden_units=(256,)
    test_preprocess=True
'
# Completed 10 trials
# test_acc           : 0.7285999417304992 +- 0.0058514890585543815
# test_cross_entropy : 1.0395817756652832 +- 0.01824528000306515
# test_loss          : 1.6023436784744263 +- 0.0171463510859092
# val_acc            : 0.7223999381065369 +- 0.004630362681456446
# val_cross_entropy  : 1.0512609720230102 +- 0.01767204124354534
# val_loss           : 1.6140228867530824 +- 0.017035035549949708
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/impl/v2.gin --bindings='
    renormalized=True
    l2_reg = 2.5e-4
    dropout_rate=0.6
    rescaled=True
    hidden_units=(256,)
    test_preprocess=True
'
# Completed 10 trials
# test_acc           : 0.7983999133110047 +- 0.0025377298113155385
# test_cross_entropy : 0.5307942867279053 +- 0.002771296155868208
# test_loss          : 0.7625577926635743 +- 0.0038078579740188612
# val_acc            : 0.816799920797348 +- 0.006881859263073673
# val_cross_entropy  : 0.5093001008033753 +- 0.003872210018369807
# val_loss           : 0.7410635948181152 +- 0.003895086859756118
```

### GCNII Heat

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/v2.gin igcn/config/impl/heat.gin --bindings='
    renormalized=True
    l2_reg = 2.5e-4
    dropout_rate=0.7
    rescaled=True
    test_preprocess=True
    t=3.05
'
# Results for 10 runs
# test_acc           : 0.837899899482727 +- 0.003884561452631169
# test_cross_entropy : 0.6327329874038696 +- 0.011478191989069315
# test_loss          : 1.0466814756393432 +- 0.006161460590554878
# val_acc            : 0.8123998939990997 +- 0.005986647622966761
# val_cross_entropy  : 0.6779792010784149 +- 0.010751146465291113
# val_loss           : 1.091927671432495 +- 0.00573599468569869
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin igcn/config/impl/v2.gin igcn/config/impl/heat.gin --bindings='
    renormalized=True
    l2_reg = 2.5e-4
    dropout_rate=0.8
    rescaled=True
    hidden_units=(256,)
    test_preprocess=True
    t=3.05
'
# Results for 10 runs
# test_acc           : 0.7156999588012696 +- 0.004754995945368579
# test_cross_entropy : 0.9542708992958069 +- 0.006566744276697956
# test_loss          : 1.4935310363769532 +- 0.01718456147825163
# val_acc            : 0.7259999573230743 +- 0.0077974359895840815
# val_cross_entropy  : 0.9669215142726898 +- 0.006242333079535732
# val_loss           : 1.5061816334724427 +- 0.01618008385768211
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/impl/v2.gin igcn/config/impl/heat.gin --bindings='
    renormalized=True
    l2_reg = 2.5e-4
    dropout_rate=0.6
    rescaled=True
    hidden_units=(256,)
    test_preprocess=True
    t=3.05
'
# Results for 10 runs
# test_acc           : 0.796499902009964 +- 0.003413228556073764
# test_cross_entropy : 0.5488942563533783 +- 0.003975265792442408
# test_loss          : 0.7596135437488556 +- 0.005770864051791691
# val_acc            : 0.7999998986721039 +- 0.004289541927403261
# val_cross_entropy  : 0.5244565367698669 +- 0.0035411263839447895
# val_loss           : 0.7351758360862732 +- 0.00570382269643143
```

### GCNII Pinv

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin igcn/config/impl/v2.gin igcn/config/impl/pinv.gin --bindings='
    l2_reg=1e-5
    dropout_rate=0.7
    test_preprocess=True
'
# Results for 10 runs
# test_acc           = 0.8163999021053314 +- 0.003720196311062873
# test_cross_entropy = 0.5898191630840302 +- 0.008842318277987283
# test_loss          = 0.7613976955413818 +- 0.006601625694277837
# val_acc            = 0.8105999052524566 +- 0.0065757288439998235
# val_cross_entropy  = 0.6795394837856292 +- 0.01273200860013438
# val_loss           = 0.851118016242981 +- 0.013247241143250425
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin igcn/config/impl/v2.gin igcn/config/impl/pinv.gin --bindings='
    l2_reg = 1e-5
    dropout_rate=0.8
    hidden_units=(256,)
    test_preprocess=True
'
# Results for 10 runs
# test_acc           = 0.5826999366283416 +- 0.005728005070346925
# test_cross_entropy = 1.2830039858818054 +- 0.01070371251399216
# test_loss          = 1.3746660232543946 +- 0.010535855758811509
# val_acc            = 0.5841999471187591 +- 0.010524252119744
# val_cross_entropy  = 1.2968915224075317 +- 0.013630229085195952
# val_loss           = 1.3885535717010498 +- 0.012873519186476765
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/impl/v2.gin igcn/config/impl/pinv.gin --bindings='
    l2_reg = 1e-5
    dropout_rate=0.6
    hidden_units=(256,)
    test_preprocess=True
'
# Results for 10 runs
# test_acc           = 0.800099891424179 +- 0.0020223841718553584
# test_cross_entropy = 0.5909909307956696 +- 0.002440003401652134
# test_loss          = 0.653499436378479 +- 0.0036616263234371645
# val_acc            = 0.8233998775482178 +- 0.004903049174378233
# val_cross_entropy  = 0.5347208261489869 +- 0.002046004788996778
# val_loss           = 0.5972293317317963 +- 0.003430191183464824
```

### DAGNN Extras

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cs/lp.gin
# Completed 10 trials
# test_acc           : 0.9328840911388397 +- 0.003659955509108191
# test_cross_entropy : 0.21307195127010345 +- 0.011594148957306365
# test_loss          : 0.21307193636894226 +- 0.011594148957306365
# val_acc            : 0.9206666707992553 +- 0.01137682863681863
# val_cross_entropy  : 0.2456260696053505 +- 0.036631772697704006
# val_loss           : 0.2456260696053505 +- 0.03663177323266801
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/physics/lp.gin
# Completed 10 trials
# test_acc           : 0.9393423616886138 +- 0.005945380796296574
# test_cross_entropy : 0.18828428089618682 +- 0.01770861113309271
# test_loss          : 0.18828428089618682 +- 0.01770861113309271
# val_acc            : 0.9486666738986969 +- 0.023485233241874136
# val_cross_entropy  : 0.17187942042946816 +- 0.06640274543558147
# val_loss           : 0.1718794196844101 +- 0.06640274603009234
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/computer/lp.gin
# Completed 10 trials
# test_acc           : 0.8287944376468659 +- 0.012897910460491569
# test_cross_entropy : 0.5862685739994049 +- 0.03057714316196658
# test_loss          : 0.719699603319168 +- 0.03283461116512767
# val_acc            : 0.8806666016578675 +- 0.025725051291343205
# val_cross_entropy  : 0.38378787636756895 +- 0.06456449204149581
# val_loss           : 0.5172190219163895 +- 0.06878198019601889
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/photo/lp.gin
# Completed 10 trials
# test_acc           : 0.9068154454231262 +- 0.014914458980882046
# test_cross_entropy : 0.4057149767875671 +- 0.02695328541886069
# test_loss          : 0.7728778004646302 +- 0.02204600996036634
# val_acc            : 0.9412499964237213 +- 0.01124998927116394
# val_cross_entropy  : 0.34073654413223264 +- 0.03537788505386753
# val_loss           : 0.7078994333744049 +- 0.03488713783044896
```

#### Renormalized and Rescaled

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cs/lp.gin --bindings='
rescaled=True
renormalized=True
l2_reg=0
'
# Results for 10 runs
# test_acc           : 0.9329636991024017 +- 0.003272456903314123
# test_cross_entropy : 0.21221044808626174 +- 0.010371370173426828
# test_loss          : 0.21221043318510055 +- 0.010371370173426828
# val_acc            : 0.9228888928890229 +- 0.01120185697882394
# val_cross_entropy  : 0.2552295535802841 +- 0.030104183028346537
# val_loss           : 0.2552295535802841 +- 0.030104183028346537
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/physics/lp.gin --bindings='
rescaled=True
renormalized=True
l2_reg=0
'
# Results for 10 runs
# test_acc           : 0.9411179006099701 +- 0.0036323178968513774
# test_cross_entropy : 0.1822742149233818 +- 0.01591691693954429
# test_loss          : 0.1822742149233818 +- 0.01591691693954429
# val_acc            : 0.9473333597183228 +- 0.027235585361703604
# val_cross_entropy  : 0.16565293595194816 +- 0.0603036001147925
# val_loss           : 0.16565293669700623 +- 0.06030359946203678
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/computer/lp.gin --bindings='
rescaled=True
renormalized=True
l2_reg=5e-6
'
# Results for 10 runs
# test_acc           : 0.825254374742508 +- 0.014810613773405692
# test_cross_entropy : 0.6001822352409363 +- 0.03970765242790043
# test_loss          : 0.7305228114128113 +- 0.03936474873759427
# val_acc            : 0.8949999451637268 +- 0.021042283341905242
# val_cross_entropy  : 0.3519410192966461 +- 0.049690850449994536
# val_loss           : 0.482281693816185 +- 0.04942336534802009
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/photo/lp.gin --bindings='
rescaled=True
renormalized=True
l2_reg=5e-5
'
# Results for 10 runs
# test_acc           : 0.9080007374286652 +- 0.01773474160982051
# test_cross_entropy : 0.4069391518831253 +- 0.02618191991723407
# test_loss          : 0.7648206353187561 +- 0.023599098822475028
# val_acc            : 0.934166669845581 +- 0.016115884302191828
# val_cross_entropy  : 0.35171073079109194 +- 0.035730299964251894
# val_loss           : 0.7095922946929931 +- 0.03540771994041499
```

With `epsilon=0.5`

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cs/lp.gin --bindings='
rescaled=True
renormalized=True
l2_reg=0
epsilon=0.5
'
# Results for 10 runs
# test_acc           : 0.9313712418079376 +- 0.0049053473263042735
# test_cross_entropy : 0.21451659947633744 +- 0.014335915438949165
# test_loss          : 0.21451658457517625 +- 0.014335915438949165
# val_acc            : 0.922222238779068 +- 0.011066568861399052
# val_cross_entropy  : 0.2537547081708908 +- 0.025065255102928775
# val_loss           : 0.2537547081708908 +- 0.025065255102928775
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/physics/lp.gin --bindings='
rescaled=True
renormalized=True
l2_reg=0
epsilon=0.5
'
# Results for 10 runs
# test_acc           : 0.9372630894184113 +- 0.003676276606102275
# test_cross_entropy : 0.1914386808872223 +- 0.015436931354375951
# test_loss          : 0.1914386808872223 +- 0.015436931354375951
# val_acc            : 0.9466666936874389 +- 0.024944371533698884
# val_cross_entropy  : 0.17972400784492493 +- 0.0610649123443331
# val_loss           : 0.1797240063548088 +- 0.06106491152068023
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/computer/lp.gin --bindings='
rescaled=True
renormalized=True
l2_reg=1.25e-5
epsilon=0.5
'
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/photo/lp.gin --bindings='
rescaled=True
renormalized=True
l2_reg=1.25e-4
epsilon=0.5
'
```

### Basic Renormalized Hyperparameter Search

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin --bindings='
renormalized=True
rescaled=True
l2_reg=1.25e-4
epsilon=0.05
'
# Results for 10 runs
# test_acc           : 0.8413998901844024 +- 0.0030066530779472955
# test_cross_entropy : 0.5688735246658325 +- 0.016039087537693363
# test_loss          : 0.9025026679039001 +- 0.007746238257210227
# val_acc            : 0.8079998791217804 +- 0.005796543936629227
# val_cross_entropy  : 0.6147442519664764 +- 0.014298094263819927
# val_loss           : 0.9483733832836151 +- 0.0070189508927600485
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin --bindings='
renormalized=True
rescaled=True
l2_reg=2.5e-4
epsilon=0.1
'
# Results for 10 runs
# test_acc           : 0.8454999089241028 +- 0.004271993613040238
# test_cross_entropy : 0.62767995595932 +- 0.010186372356622944
# test_loss          : 1.0740058660507201 +- 0.0061913400569793674
# val_acc            : 0.816999888420105 +- 0.004837351514494726
# val_cross_entropy  : 0.6693135976791382 +- 0.008840258182365817
# val_loss           : 1.1156394839286805 +- 0.006657431539444735
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/cora/lp.gin --bindings='
renormalized=True
rescaled=True
l2_reg=5e-4
epsilon=0.2
'
# Results for 10 runs
# test_acc           : 0.8428999185562134 +- 0.0029816112981376377
# test_cross_entropy : 0.8260805904865265 +- 0.005419145570563039
# test_loss          : 1.3820869684219361 +- 0.004196384632836031
# val_acc            : 0.8121999025344848 +- 0.007560428735586892
# val_cross_entropy  : 0.8608231365680694 +- 0.005922865558753183
# val_loss           : 1.416829514503479 +- 0.005228634615693692

python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin --bindings='
renormalized=True
rescaled=True
l2_reg=5e-4
epsilon=0.05
'
# Results for 10 runs
# test_acc           : 0.7252999305725097 +- 0.003742974724985947
# test_cross_entropy : 0.9719705045223236 +- 0.0051464395439807156
# test_loss          : 1.5149131655693053 +- 0.003607998496207698
# val_acc            : 0.7347999393939972 +- 0.006399966403936245
# val_cross_entropy  : 1.0018361747264861 +- 0.0036251982077060286
# val_loss           : 1.5447788119316102 +- 0.002739175929948438
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin --bindings='
renormalized=True
rescaled=True
l2_reg=1e-3
epsilon=0.1
'
# Results for 10 runs
# test_acc           : 0.7281999468803406 +- 0.004261448121346184
# test_cross_entropy : 1.170055079460144 +- 0.0024878989646006273
# test_loss          : 1.7881271600723267 +- 0.00649750764775896
# val_acc            : 0.73499995470047 +- 0.004404556771642058
# val_cross_entropy  : 1.1936612963676452 +- 0.003517095364794136
# val_loss           : 1.8117334008216859 +- 0.0067637403350219415
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/citeseer/lp.gin --bindings='
renormalized=True
rescaled=True
l2_reg=2e-3
epsilon=0.2
'
# Results for 10 runs
# test_acc           : 0.6425999343395233 +- 0.026593977818679213
# test_cross_entropy : 1.5263304710388184 +- 0.005631595382701639
# test_loss          : 1.902407467365265 +- 0.004907610614953865
# val_acc            : 0.6223999321460724 +- 0.03564042365515103
# val_cross_entropy  : 1.535089135169983 +- 0.004795558589862625
# val_loss           : 1.9111661434173584 +- 0.005832445545507685

python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin --bindings='
renormalized=True
rescaled=True
l2_reg=1.25e-4
epsilon=0.05
'
# Results for 10 runs
# test_acc           : 0.8046998500823974 +- 0.0029342888899007474
# test_cross_entropy : 0.5173466801643372 +- 0.003295219864980752
# test_loss          : 0.6714585721492767 +- 0.0035389198321745265
# val_acc            : 0.8251998960971832 +- 0.006462188359113921
# val_cross_entropy  : 0.48882146179676056 +- 0.004112174590960572
# val_loss           : 0.6429333448410034 +- 0.003680238604213048
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin --bindings='
renormalized=True
rescaled=True
l2_reg=2.5e-4
epsilon=0.1
'
# Results for 10 runs
# test_acc           : 0.8008999109268189 +- 0.002624882352304222
# test_cross_entropy : 0.5158469617366791 +- 0.004058498741856773
# test_loss          : 0.7169864475727081 +- 0.005435798483695574
# val_acc            : 0.8159999012947082 +- 0.009252018497016378
# val_cross_entropy  : 0.49418518543243406 +- 0.004864599164388392
# val_loss           : 0.6953246712684631 +- 0.0038375270596445956
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin --bindings='
renormalized=True
rescaled=True
l2_reg=5e-4
epsilon=0.2
'
# Results for 10 runs
# test_acc           : 0.7950998902320862 +- 0.004948730228895951
# test_cross_entropy : 0.5435547530651093 +- 0.0040406570580601014
# test_loss          : 0.8088864922523499 +- 0.0034834220282970703
# val_acc            : 0.80779989361763 +- 0.0048538724345445405
# val_cross_entropy  : 0.5285917758941651 +- 0.003338754936231966
# val_loss           : 0.7939235270023346 +- 0.0020917311674402287

```

### Pubmed implementation variants with performance speeds/memory requirements

Results (`test_acc` etc) are from the same commands without `igcn/config/utils/time.gin`.

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/low-rank.gin --bindings='
rescaled=False
l2_reg=2.5e-3
rank=32
'
# Mean time: 0.021380081176757812
# Completed 10 trials
# test_acc           : 0.7284999489784241 +- 0.005572264914839581
# test_cross_entropy : 0.7055353462696076 +- 0.003022909030501598
# test_loss          : 0.7977785706520081 +- 0.003486273995768541
# val_acc            : 0.7331999599933624 +- 0.006823496732001593
# val_cross_entropy  : 0.6896505653858185 +- 0.0028626156074750813
# val_loss           : 0.7818938016891479 +- 0.0030441160657586966
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/low-rank.gin --bindings='
rescaled=False
l2_reg=2.5e-3
rank=64
'
# Mean time: 0.022319774627685546
# Completed 10 trials
# test_acc           : 0.7689998865127563 +- 0.003065929687900925
# test_cross_entropy : 0.6601273477077484 +- 0.0028111447426510837
# test_loss          : 0.7775860249996185 +- 0.007816449699649758
# val_acc            : 0.7807998895645142 +- 0.005230677025753289
# val_cross_entropy  : 0.6170040249824524 +- 0.002140174280914854
# val_loss           : 0.7344627022743225 +- 0.006198633778080688
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/low-rank.gin --bindings='
rescaled=False
l2_reg=2.5e-3
rank=128
'
# Mean time: 0.02263925313949585
# Completed 10 trials
# test_acc           : 0.7591998875141144 +- 0.006029915747357829
# test_cross_entropy : 0.6641319453716278 +- 0.004983714130936896
# test_loss          : 0.8114212155342102 +- 0.012235645837964918
# val_acc            : 0.7835998892784118 +- 0.010762896682288177
# val_cross_entropy  : 0.5901943147182465 +- 0.0028442974765073613
# val_loss           : 0.7374835729598999 +- 0.00826742298410412
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/low-rank.gin --bindings='
rescaled=False
l2_reg=2.5e-3
rank=256
'
# Mean time: 0.025996685028076172
# Completed 10 trials
# test_acc           : 0.787799882888794 +- 0.001661315582471623
# test_cross_entropy : 0.6114749610424042 +- 0.005261709543529513
# test_loss          : 0.7852826654911041 +- 0.008548913393578881
# val_acc            : 0.8011999011039734 +- 0.003919177539275522
# val_cross_entropy  : 0.5496959686279297 +- 0.004137362038537428
# val_loss           : 0.7235036551952362 +- 0.005173766064257835
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/low-rank.gin --bindings='
rescaled=False
l2_reg=2.5e-3
rank=512
'
# Mean time: 0.031128764152526855
# Completed 10 trials
# test_acc           : 0.7965999186038971 +- 0.003136878465279187
# test_cross_entropy : 0.5870552659034729 +- 0.008852233664478218
# test_loss          : 0.7808372795581817 +- 0.014930146491184567
# val_acc            : 0.8191998898983002 +- 0.006462211233649413
# val_cross_entropy  : 0.5333521485328674 +- 0.0024190158120679416
# val_loss           : 0.7271341621875763 +- 0.008535658881400508

python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=1e-0
'
# Mean time: 0.023210828304290772
# Completed 10 trials
# test_acc           : 0.7646998941898346 +- 0.0078108901534819355
# test_cross_entropy : 0.6140918493270874 +- 0.01817877081881146
# test_loss          : 0.8526180267333985 +- 0.024956146693674033
# val_acc            : 0.783399897813797 +- 0.009254188252430353
# val_cross_entropy  : 0.5973521411418915 +- 0.017488013748707388
# val_loss           : 0.835878312587738 +- 0.023230928102031343
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=5e-1
'
# Mean time: 0.025132465362548827
# Completed 10 trials
# test_acc           : 0.789099907875061 +- 0.005906759360061159
# test_cross_entropy : 0.550175404548645 +- 0.008017011401030852
# test_loss          : 0.7709007561206818 +- 0.013417931955726378
# val_acc            : 0.8015998899936676 +- 0.00685858656967112
# val_cross_entropy  : 0.5224186778068542 +- 0.009858462097889119
# val_loss           : 0.7431440234184266 +- 0.014526762042821077
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=2e-1
'
# Mean time: 0.028145329952239992
# Completed 10 trials
# test_acc           : 0.8038998901844024 +- 0.002624883033460928
# test_cross_entropy : 0.5252127408981323 +- 0.008190838639261427
# test_loss          : 0.7243494331836701 +- 0.011297937636739997
# val_acc            : 0.820599889755249 +- 0.006327725392525435
# val_cross_entropy  : 0.49007178843021393 +- 0.00446770856674691
# val_loss           : 0.6892084836959839 +- 0.008519845544419235
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=1e-1
'
# Mean time: 0.02915379047393799
# Completed 10 trials
# test_acc           : 0.8075998902320862 +- 0.003826242083303021
# test_cross_entropy : 0.5180291533470154 +- 0.006970822577042231
# test_loss          : 0.7259968221187592 +- 0.012099027545583445
# val_acc            : 0.8243999123573303 +- 0.009624949881747684
# val_cross_entropy  : 0.4839375078678131 +- 0.0039354971371690795
# val_loss           : 0.6919051766395569 +- 0.008696636520448527
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=5e-2
'
# Mean time: 0.031570730209350584
# Completed 10 trials
# test_acc           : 0.8049998760223389 +- 0.0037416681555481114
# test_cross_entropy : 0.5159908652305603 +- 0.00628625293707876
# test_loss          : 0.7169031858444214 +- 0.00735114393198308
# val_acc            : 0.8239999055862427 +- 0.006387478318198003
# val_cross_entropy  : 0.4846501499414444 +- 0.0043492581862632955
# val_loss           : 0.6855624735355377 +- 0.005520822984000393
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=2e-2
'
# Mean time: 0.03273360252380371
# Completed 10 trials
# test_acc           : 0.8030998885631562 +- 0.003645547995817191
# test_cross_entropy : 0.5169268131256104 +- 0.006562883429201425
# test_loss          : 0.7238783121109009 +- 0.012097619646732725
# val_acc            : 0.819999897480011 +- 0.007429661237543621
# val_cross_entropy  : 0.4817113071680069 +- 0.004200259731363662
# val_loss           : 0.6886628031730652 +- 0.007492083808595189
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=1e-2
'
# Mean time: 0.035237772464752196
# Completed 10 trials
# test_acc           : 0.8024998903274536 +- 0.004318542791380819
# test_cross_entropy : 0.5216238975524903 +- 0.008146002874504744
# test_loss          : 0.7277401864528656 +- 0.01394051371029202
# val_acc            : 0.8185999095439911 +- 0.00800252037057082
# val_cross_entropy  : 0.4833801567554474 +- 0.002484631116377632
# val_loss           : 0.6894964516162873 +- 0.0066258170797119315
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=5e-3
'
# Mean time: 0.03685513734817505
# Completed 10 trials
# test_acc           : 0.8026999056339263 +- 0.0038999840236146922
# test_cross_entropy : 0.5204010367393493 +- 0.006611292449480389
# test_loss          : 0.7268378257751464 +- 0.01332889254469403
# val_acc            : 0.8167999029159546 +- 0.006144936035324939
# val_cross_entropy  : 0.4824660927057266 +- 0.0022229471299152893
# val_loss           : 0.6889028787612915 +- 0.007044518554199768
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=2e-3
'
# Mean time: 0.038984262943267824
# Completed 10 trials
# test_acc           : 0.8012998998165131 +- 0.003067573252041491
# test_cross_entropy : 0.5190572738647461 +- 0.005130542054319834
# test_loss          : 0.7242738544940949 +- 0.011707694531054453
# val_acc            : 0.8181999146938324 +- 0.006477674845003565
# val_cross_entropy  : 0.4822594225406647 +- 0.0025936705199118364
# val_loss           : 0.6874760031700134 +- 0.007631606758831835
python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin igcn/config/impl/lazy.gin --bindings='
rescaled=False
l2_reg=2.5e-3
tol=1e-3
'
# Mean time: 0.041272776126861574
# Completed 10 trials
# test_acc           : 0.8037999093532562 +- 0.0017204539898124645
# test_cross_entropy : 0.5166951179504394 +- 0.006832513304864417
# test_loss          : 0.7211368620395661 +- 0.012334438807139553
# val_acc            : 0.8213999032974243 +- 0.008946510306000304
# val_cross_entropy  : 0.48196640610694885 +- 0.002759679036494332
# val_loss           : 0.6864081561565399 +- 0.007449895323541036


python -m graph_tf gtf_config/build_and_fit_many.gin igcn/config/pubmed/lp.gin igcn/config/utils/time.gin --bindings='
rescaled=False
l2_reg=2.5e-3
'
# Mean time: 0.020820167064666748
# Completed 10 trials
# test_acc           : 0.8046999096870422 +- 0.0017349090417887252
# test_cross_entropy : 0.5168011784553528 +- 0.00642947139978373
# test_loss          : 0.7212016105651855 +- 0.011853680718952746
# val_acc            : 0.8213999032974243 +- 0.008534646118343833
# val_cross_entropy  : 0.481717512011528 +- 0.0031725474953642955
# val_loss           : 0.6861179530620575 +- 0.007459909812384457
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/pubmed.gin igcn/config/utils/time.gin
# Mean time: 0.004775912761688233
# Completed 10 trials
# test_acc           : 0.7936999022960662 +- 0.004148498338249456
# test_cross_entropy : 0.5568033576011657 +- 0.005979646246428646
# test_loss          : 0.8921212315559387 +- 0.010446776093484737
# val_acc            : 0.8061998784542084 +- 0.008316256618820787
# val_cross_entropy  : 0.530375337600708 +- 0.005684673435508793
# val_loss           : 0.8656932234764099 +- 0.009103470139997584

python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/pubmed.gin igcn/config/utils/time.gin
# Mean time: 0.034050731658935546
# Completed 10 trials
# test_acc           : 0.8051999092102051 +- 0.004044752427805108
# test_cross_entropy : 0.5286962032318115 +- 0.011760741322258812
# test_loss          : 0.756601768732071 +- 0.013891932374749668
# val_acc            : 0.8200000286102295 +- 0.008390476423019579
# val_cross_entropy  : 0.4865717262029648 +- 0.006494213652244235
# val_loss           : 0.7144772350788117 +- 0.006707570416937157
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/pubmed.gin igcn/config/utils/time.gin --bindings='simplified=True'
# Mean time: 0.02358098030090332
# Completed 10 trials
# test_acc           : 0.8022998869419098 +- 0.0018466263891035975
# test_cross_entropy : 0.528484684228897 +- 0.007900193507670211
# test_loss          : 0.7285901963710785 +- 0.01751516394568192
# val_acc            : 0.8166000187397003 +- 0.0076446227592287625
# val_cross_entropy  : 0.4888902485370636 +- 0.005028369217080441
# val_loss           : 0.6889957010746002 +- 0.013688193497584213


python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/pubmed.gin igcn/config/utils/time.gin
# Mean time: 0.39329115867614745
# Completed 10 trials
# test_acc           : 0.7961999237537384 +- 0.0022271107938280324
# test_cross_entropy : 0.5714908361434936 +- 0.010106126690822954
# test_loss          : 0.7949661314487457 +- 0.007666459425290565
# val_acc            : 0.8004000186920166 +- 0.007088013586322617
# val_cross_entropy  : 0.515491396188736 +- 0.007640654792928717
# val_loss           : 0.7389666378498078 +- 0.006210819260865943
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/pubmed.gin igcn/config/utils/time.gin --bindings='simplified=True'
# Mean time: 0.2599149513244629
# Results for 10 runs
# test_acc           : 0.7994998753070831 +- 0.0037749144071941437
# test_cross_entropy : 0.5219661831855774 +- 0.004417817778264475
# test_loss          : 0.7225155889987945 +- 0.0041216240420093465
# val_acc            : 0.8098000466823578 +- 0.005095108931348702
# val_cross_entropy  : 0.5054759919643402 +- 0.0045692479882353644
# val_loss           : 0.7060253500938416 +- 0.003352012263006577

```
