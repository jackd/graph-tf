# MLP

Simple baseline implementation.

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin \
    mlp/config/cora.gin
# Results for 10 runs
# test_acc           = 0.40449994504451753 +- 0.036491764360653774
# test_cross_entropy = 1.8975337266921997 +- 0.0022065481568414843
# test_loss          = 2.0218807220458985 +- 0.0028595270164007173
# val_acc            = 0.4223999410867691 +- 0.04239622126039275
# val_cross_entropy  = 1.8964861154556274 +- 0.0019892215593025142
# val_loss           = 2.020833134651184 +- 0.0031226459286720475
python -m graph_tf gtf_config/build_and_fit_many.gin \
    mlp/config/cora.gin \
    mlp/config/utils/spectral.gin
# Results for 10 runs
# test_acc           = 0.7075999677181244 +- 0.009297315599844648
# test_cross_entropy = 1.6915923237800599 +- 0.003972169528529686
# test_loss          = 2.2295357465744017 +- 0.004782917854833981
# val_acc            = 0.7005999445915222 +- 0.014256216090877847
# val_cross_entropy  = 1.6992536187171936 +- 0.0037449545650977923
# val_loss           = 2.237197017669678 +- 0.00501114746634644
python -m graph_tf gtf_config/build_and_fit_many.gin \
    mlp/config/cora.gin \
    mlp/config/utils/page-rank.gin
# Results for 10 runs
# test_acc           = 0.8419998824596405 +- 0.0053291624162713335
# test_cross_entropy = 0.821976363658905 +- 0.004271974300808684
# test_loss          = 1.5087429761886597 +- 0.007532387495293216
# val_acc            = 0.8169999122619629 +- 0.0054589132026436156
# val_cross_entropy  = 0.8476366221904754 +- 0.003689180280873769
# val_loss           = 1.5344032287597655 +- 0.008262543012008704
python -m graph_tf gtf_config/build_and_fit_many.gin \
    mlp/config/cora.gin \
    mlp/config/utils/page-rank.gin \
    --bindings='
      include_transformed_features=False
'
# Results for 10 runs
# test_acc           = 0.8418999135494232 +- 0.005629396654008932
# test_cross_entropy = 0.8272308349609375 +- 0.00468111704822371
# test_loss          = 1.472713792324066 +- 0.008147042559663736
# val_acc            = 0.8177998960018158 +- 0.006539092524797222
# val_cross_entropy  = 0.8524779677391052 +- 0.005295936912797639
# val_loss           = 1.4979609370231628 +- 0.008601211835403333
python -m graph_tf gtf_config/build_and_fit.gin \
    mlp/config/ogbn-arxiv.gin
# Final results
# test_acc           : 0.5477234125137329
# test_cross_entropy : 1.5347206592559814
# test_loss          : 1.534720778465271
# val_acc            : 0.5715292692184448
# val_cross_entropy  : 1.48497474193573
# val_loss           : 1.4849746227264404
python -m graph_tf gtf_config/build_and_fit.gin \
    mlp/config/ogbn-arxiv.gin \
    mlp/config/utils/spectral.gin
# Final results
# test_acc           : 0.6093450784683228
# test_cross_entropy : 1.296971321105957
# test_loss          : 1.296971321105957
# val_acc            : 0.6263633370399475
# val_cross_entropy  : 1.2771133184432983
# val_loss           : 1.2771131992340088
python -m graph_tf gtf_config/build_and_fit.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin
# Final results
# test_acc           : 0.7021788954734802
# test_cross_entropy : 0.9461618661880493
# test_loss          : 0.9461618661880493
# val_acc            : 0.7152924537658691
# val_cross_entropy  : 0.9119006991386414
# val_loss           : 0.9119006395339966
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
# Results for 10 runs
# test_acc           = 0.6942184746265412 +- 0.004496337727582618
# test_cross_entropy = 0.9782700657844543 +- 0.01961790036160597
# test_loss          = 0.9782700598239898 +- 0.01961790574852028
# val_acc            = 0.7052653074264527 +- 0.005533138624581985
# val_cross_entropy  = 0.9455797135829925 +- 0.020290075153031193
# val_loss           = 0.9455796599388122 +- 0.02029006773100859
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='include_transformed_features=False'  # only use page-rank features
# Results for 10 runs
# test_acc           = 0.7111371874809265 +- 0.0015090464151922215
# test_cross_entropy = 0.9080274224281311 +- 0.0030780089881361824
# test_loss          = 0.9080274045467377 +- 0.0030780081438958
# val_acc            = 0.7238162696361542 +- 0.0008562293914665742
# val_cross_entropy  = 0.8716644704341888 +- 0.001995269126270907
# val_loss           = 0.8716644048690796 +- 0.0019952694662793667
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512]
'
# Results for 10 runs
# test_acc           = 0.7132070124149322 +- 0.0016861090881623388
# test_cross_entropy = 0.9032229304313659 +- 0.004059384889170944
# test_loss          = 0.9032229423522949 +- 0.004059369701388178
# val_acc            = 0.7259841203689575 +- 0.0015990856471133094
# val_cross_entropy  = 0.8652531981468201 +- 0.002554183104232118
# val_loss           = 0.8652531385421753 +- 0.002554183104232118
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[1024]
'
# Results for 10 runs
# test_acc           = 0.7139188885688782 +- 0.00241969067380774
# test_cross_entropy = 0.9007848381996155 +- 0.00367447147269233
# test_loss          = 0.9007848262786865 +- 0.00367445305334815
# val_acc            = 0.7266150116920471 +- 0.001337644353700605
# val_cross_entropy  = 0.8640537440776825 +- 0.0017659409277477542
# val_loss           = 0.8640536963939667 +- 0.0017659397478689816
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512]
'
# Results for 10 runs
# test_acc           = 0.715052580833435 +- 0.002385590733175804
# test_cross_entropy = 0.8779021143913269 +- 0.004943536933143535
# test_loss          = 0.8779021143913269 +- 0.0049435505788641195
# val_acc            = 0.7283365368843079 +- 0.0010241621266698347
# val_cross_entropy  = 0.8384712040424347 +- 0.0034400108254561876
# val_loss           = 0.8384711503982544 +- 0.003440010465326772
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512]
     dropout_rate=0.4
'
# Results for 10 runs
# test_acc           = 0.7164681434631348 +- 0.0027235253630347846
# test_cross_entropy = 0.8761351525783538 +- 0.004190477594003264
# test_loss          = 0.8761351644992829 +- 0.004190478652099416
# val_acc            = 0.7303600907325745 +- 0.0016238979083632978
# val_cross_entropy  = 0.8339183449745178 +- 0.002884172325897951
# val_loss           = 0.833918285369873 +- 0.0028841624627624915
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512]
     dropout_rate=0.4
     epsilon=[0.1,0.5]
'
# Results for 10 runs
# test_acc           = 0.7170668721199036 +- 0.0021450909732463035
# test_cross_entropy = 0.8807851016521454 +- 0.004129501353084737
# test_loss          = 0.8807851076126099 +- 0.004129489606838613
# val_acc            = 0.7292761743068695 +- 0.0014278443366072255
# val_cross_entropy  = 0.843551081418991 +- 0.001954258860982287
# val_loss           = 0.8435510277748108 +- 0.0019542626691413257
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512]
     dropout_rate=0.4
     epsilon=[0.1,0.5]
     lr=5e-3
'
# Results for 10 runs
# test_acc           = 0.7177787542343139 +- 0.0013326619568981791
# test_cross_entropy = 0.8730893075466156 +- 0.0019544777402345165
# test_loss          = 0.8730893015861512 +- 0.0019544845046067104
# val_acc            = 0.730034589767456 +- 0.0007446599445043222
# val_cross_entropy  = 0.8351155698299408 +- 0.0017396348261426472
# val_loss           = 0.835115498304367 +- 0.0017396270815035063
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512]
     dropout_rate=0.4
     epsilon=[0.1,0.5]
     lr=2e-3
'
# Results for 10 runs
# test_acc           = 0.7197354137897491 +- 0.001566785231745575
# test_cross_entropy = 0.8663702547550202 +- 0.003503890062793517
# test_loss          = 0.8663702726364135 +- 0.0035038612075647133
# val_acc            = 0.7321688830852509 +- 0.0011841283304735402
# val_cross_entropy  = 0.8260863780975342 +- 0.0014172686859528234
# val_loss           = 0.8260863244533538 +- 0.0014172697056536946
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512]
     dropout_rate=0.5
     epsilon=[0.1,0.5]
     lr=2e-3
'
# Results for 10 runs
# test_acc           = 0.7204123437404633 +- 0.0014952772661190406
# test_cross_entropy = 0.8649904370307923 +- 0.0020334129314276686
# test_loss          = 0.8649904370307923 +- 0.0020333969629497796
# val_acc            = 0.7327293038368226 +- 0.0008331916650753708
# val_cross_entropy  = 0.8248302519321442 +- 0.0010446624740134236
# val_loss           = 0.8248301863670349 +- 0.001044661723366178
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[1024,512]
     dropout_rate=0.5
     epsilon=[0.1,0.5]
     lr=2e-3
'
# Results for 10 runs
# test_acc           = 0.7217929124832153 +- 0.002009630193382498
# test_cross_entropy = 0.8616129457950592 +- 0.0035653661231561636
# test_loss          = 0.8616129398345947 +- 0.0035653618144931925
# val_acc            = 0.7342964947223664 +- 0.0007952895603178687
# val_cross_entropy  = 0.8221681714057922 +- 0.0007448821026928211
# val_loss           = 0.8221681177616119 +- 0.000744896477081744
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[1024,512]
     dropout_rate=0.5
     epsilon=[0.1]
     lr=2e-3
'
# Results for 10 runs
# test_acc           = 0.7222887516021729 +- 0.0017978015781232365
# test_cross_entropy = 0.856004822254181 +- 0.004058884047357077
# test_loss          = 0.856004822254181 +- 0.004058870161331598
# val_acc            = 0.7360884845256805 +- 0.0012321898825259218
# val_cross_entropy  = 0.8110549807548523 +- 0.001005060126886721
# val_loss           = 0.811054915189743 +- 0.001005057574756057
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[1024,1024]
     dropout_rate=0.6
     epsilon=[0.1,0.5]
     lr=2e-3
'
# Results for 10 runs
# test_acc           = 0.7217249989509582 +- 0.0011024692147818657
# test_cross_entropy = 0.8596224844455719 +- 0.0024147938942711564
# test_loss          = 0.8596224665641785 +- 0.0024148011946947185
# val_acc            = 0.7344843983650208 +- 0.0009908897522604902
# val_cross_entropy  = 0.819694995880127 +- 0.0008992621780180423
# val_loss           = 0.8196949362754822 +- 0.0008992626927939011
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512]
     dropout_rate=0.3
'
# Results for 10 runs
# test_acc           = 0.7164928138256073 +- 0.001530549869195742
# test_cross_entropy = 0.8790741384029388 +- 0.0035319306735967257
# test_loss          = 0.8790741443634034 +- 0.0035319225377686623
# val_acc            = 0.7299674868583679 +- 0.001770480710641846
# val_cross_entropy  = 0.8382565200328826 +- 0.003117260684428953
# val_loss           = 0.8382564604282379 +- 0.003117260684428953
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512]
     dropout_rate=0.3
     input_dropout_rate=0.1
'


python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512,512]
'
# Results for 10 runs
# test_acc           = 0.7139127194881439 +- 0.003015054003074061
# test_cross_entropy = 0.8913200974464417 +- 0.005380311662175783
# test_loss          = 0.8913201093673706 +- 0.005380298446447244
# val_acc            = 0.7272022724151611 +- 0.002377628613546244
# val_cross_entropy  = 0.8492718517780304 +- 0.004454375757852869
# val_loss           = 0.8492717862129211 +- 0.004454378394546424
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512,512]
     epsilon=[0.1, 0.5]
'
# Results for 10 runs
# test_acc           = 0.7161306977272034 +- 0.0016592270954411017
# test_cross_entropy = 0.883070957660675 +- 0.0031260709631390484
# test_loss          = 0.883070957660675 +- 0.0031260655271284636
# val_acc            = 0.7285412549972534 +- 0.0013693001544979719
# val_cross_entropy  = 0.8461374700069427 +- 0.002813255667133585
# val_loss           = 0.846137410402298 +- 0.002813255667133585
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512]
     dropout_rate=0.6
'
# Results for 10 runs
# test_acc           = 0.7130136013031005 +- 0.0019551638936209
# test_cross_entropy = 0.903610634803772 +- 0.0033464174644273195
# test_loss          = 0.903610622882843 +- 0.0033464184532875484
# val_acc            = 0.7249404728412628 +- 0.001502287096791554
# val_cross_entropy  = 0.8678951799869538 +- 0.0017021162038516739
# val_loss           = 0.867895120382309 +- 0.0017021162038516739
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[512]
     dropout_rate=0.7
'
# Results for 10 runs
# test_acc           = 0.710540509223938 +- 0.0019511826708690507
# test_cross_entropy = 0.9089025974273681 +- 0.0037586832445867836
# test_loss          = 0.9089025974273681 +- 0.0037586886204174362
# val_acc            = 0.7237390875816345 +- 0.0010308377176005766
# val_cross_entropy  = 0.871871417760849 +- 0.0029691536147610025
# val_loss           = 0.8718713581562042 +- 0.0029691536147610025
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin \
   --bindings='
     include_transformed_features=False
     units=[256,256]
'
# Results for 10 runs
# test_acc           = 0.7123326003551483 +- 0.0023196205281517893
# test_cross_entropy = 0.8878093540668488 +- 0.004060969179989688
# test_loss          = 0.8878093481063842 +- 0.004060974466120509
# val_acc            = 0.7250512123107911 +- 0.001035069909608897
# val_cross_entropy  = 0.8500521779060364 +- 0.002820715101716653
# val_loss           = 0.8500521242618561 +- 0.002820710323321287


python -m graph_tf gtf_config/build_and_fit_many.gin \
    config/ogbn-arxiv.gin \
    config/utils/page-rank.gin \
    igcn/config/losses/quad.gin
# Results for 10 runs
# test_acc                = 0.5515420973300934 +- 0.19170338595876152
# test_cross_entropy      = 2.762668490409851 +- 0.04226119911152964
# test_loss               = -6.893861734867096 +- 3.6577445709950704
# test_quad_cross_entropy = -6.893861734867096 +- 3.6577445709950704
# val_acc                 = 0.5655022293329239 +- 0.18760663171944575
# val_cross_entropy       = 2.753391981124878 +- 0.023721234756538038
# val_loss                = -7.127738726139069 +- 3.7778503030988353
# val_quad_cross_entropy  = -7.127739346027374 +- 3.7778506032121455
python -m graph_tf gtf_config/build_and_fit_many.gin \
    mlp/config/ogbn-arxiv.gin \
    mlp/config/utils/page-rank.gin igcn/config/losses/quad.gin \
    --bindings="monitor='val_quad_cross_entropy'"
# Results for 10 runs
# test_acc                = 0.6886056661605835 +- 0.004370978787818776
# test_cross_entropy      = 3.0518511295318604 +- 0.04398409811811501
# test_loss               = -9.97844820022583 +- 0.06660351909784977
# test_quad_cross_entropy = -9.97844820022583 +- 0.06660351909784977
# val_acc                 = 0.6994798898696899 +- 0.0031420928092466575
# val_cross_entropy       = 3.015373182296753 +- 0.030108230313669056
# val_loss                = -10.248607158660889 +- 0.06030754986587472
# val_quad_cross_entropy  = -10.248608207702636 +- 0.060307483367854076
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/ogbn-arxiv.gin \
   mlp/config/utils/page-rank.gin igcn/config/losses/quad.gin \
   --bindings="
monitor='val_acc'
mode='max'
"
# Results for 10 runs
# test_acc                = 0.6941258907318115 +- 0.0021165803618628663
# test_cross_entropy      = 2.8227633237838745 +- 0.034013222004606505
# test_loss               = -9.800707626342774 +- 0.08685750726054473
# test_quad_cross_entropy = -9.800707626342774 +- 0.08685750726054473
# val_acc                 = 0.7036142468452453 +- 0.0017190754197330639
# val_cross_entropy       = 2.809351372718811 +- 0.033015391510677554
# val_loss                = -10.074717235565185 +- 0.07860383131779795
# val_quad_cross_entropy  = -10.074718284606934 +- 0.0786036238248599
```

## bojchevski

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin \
   mlp/config/bojchevski/cora-full.gin \
   --bindings='
      units=[512,512]
      epsilon=[0.1]
      lr=2e-3
      dropout_rate=0.5
'
# Results for 10 runs
# test_acc           = 0.6389705300331116 +- 0.008230289947913125
# test_cross_entropy = 1.4770480632781982 +- 0.052321780195714214
# test_loss          = 1.6914951801300049 +- 0.054189979661840144
# val_acc            = 0.6385785460472106 +- 0.0037275100422186594
# val_cross_entropy  = 1.4600697040557862 +- 0.022291894548812193
# val_loss           = 1.6745166659355164 +- 0.023796448878388166
python -m graph_tf gtf_config/build_and_fit_many.gin \
      mlp/config/bojchevski/cora-full.gin
# Results for 10 runs
# test_acc           = 0.6379412293434144 +- 0.009063415617207174
# test_cross_entropy = 1.4165821433067323 +- 0.02719184278581764
# test_loss          = 0.3932408273220062 +- 0.00528616356345825
# val_acc            = 0.6434712290763855 +- 0.00385953972483396
# val_cross_entropy  = 1.398248827457428 +- 0.011814053421795976
# val_loss           = 0.238646699488163 +- 0.0024360345200215164
python -m graph_tf gtf_config/build_and_fit_many.gin \
      mlp/config/bojchevski/pubmed.gin
# Results for 10 runs
# test_acc           = 0.766495269536972 +- 0.017214122850358487
# test_cross_entropy = 0.6959484100341797 +- 0.08038682153097022
# test_loss          = 0.727071076631546 +- 0.08066899631385835
# val_acc            = 0.7601666271686554 +- 0.016473058504151473
# val_cross_entropy  = 0.725170087814331 +- 0.08807343578070642
# val_loss           = 0.756292748451233 +- 0.08835532169145614
python -m graph_tf gtf_config/build_and_fit_many.gin \
      mlp/config/bojchevski/pubmed-v2.gin
# Results for 10 runs
# test_acc           = 0.7810830712318421 +- 0.01573457033973163
# test_cross_entropy = 0.5830625116825103 +- 0.05453872343111276
# test_loss          = 0.8744827508926392 +- 0.05927637502453266
# val_acc            = 0.7769999623298645 +- 0.019189103334667054
# val_cross_entropy  = 0.5993338227272034 +- 0.05967085195755746
# val_loss           = 0.8907540440559387 +- 0.06262544451990841
python -m graph_tf gtf_config/build_and_fit_many.gin \
      mlp/config/bojchevski/reddit.gin
# Results for 10 runs
# test_acc           = 0.3356039226055145 +- 0.01027935113976496
# test_cross_entropy = 2.649855399131775 +- 0.04183586693994572
# test_loss          = 0.1456687033176422 +- 0.01052559626588858
# val_acc            = 0.3371219515800476 +- 0.011472389201054548
# val_cross_entropy  = 2.6513593435287475 +- 0.04563863407628565
# val_loss           = 0.305010986328125 +- 0.009560410349190316
python -m graph_tf gtf_config/build_and_fit_many.gin \
      mlp/config/bojchevski/mag-coarse.gin
# Results for 10 runs
# test_acc           = 0.6162977993488312 +- 0.016971427204602776
# test_cross_entropy = 1.6589121699333191 +- 0.09144597003164721
# test_loss          = 0.008323997678235173 +- 0.0003652826421255878
# val_acc            = 0.6192499160766601 +- 0.022796306452700324
# val_cross_entropy  = 1.6649020671844483 +- 0.07138547872649152
# val_loss           = 0.5211715191602707 +- 0.022844412032875697
```
