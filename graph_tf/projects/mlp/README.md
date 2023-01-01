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
      show_progress=False
'
# Results for 10 runs
# test_acc           = 0.8442999362945557 +- 0.0027221207764667943
# test_cross_entropy = 0.8388396263122558 +- 0.006851345647939745
# test_loss          = 1.40644029378891 +- 0.004432039677631697
# val_acc            = 0.8149998843669891 +- 0.0030000070732634646
# val_cross_entropy  = 0.8651405036449432 +- 0.005754320652823371
# val_loss           = 1.4327411770820617 +- 0.003707682757539922
python -m graph_tf gtf_config/build_and_fit.gin \
    mlp/config/ogbn-arxiv.gin
# Final results
# test_acc           : 0.5477234125137329
# test_cross_entropy : 1.

5347206592559814
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
   mlp/config/utils/page-rank.gin
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
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/ogbn-papers100m.gin
# Results for 10 runs
# test_acc           = 0.6621457695960998 +- 0.001861402750747849
# test_cross_entropy = 1.1687959909439087 +- 0.008135173889988532
# test_loss          = 1.1687959790229798 +- 0.008135165647139862
# val_acc            = 0.6937867701053619 +- 0.0018579952999609741
# val_cross_entropy  = 0.9806791305541992 +- 0.005602744690364286
# val_loss           = 0.9806791305541992 +- 0.005602744690364286
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/ogbn-papers100m.gin --bindings='epsilon=0.1'
# Results for 10 runs
# test_acc           = 0.6522702574729919 +- 0.0011449370265857333
# test_cross_entropy = 1.1991825580596924 +- 0.003307155694566811
# test_loss          = 1.1991825461387635 +- 0.0033071551599552613
# val_acc            = 0.6821793854236603 +- 0.0013268127732730041
# val_cross_entropy  = 1.0192213296890258 +- 0.003535148719091241
# val_loss           = 1.0192213177680969 +- 0.0035351621034998386
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/ogbn-papers100m.gin --bindings='epsilon=1.0'
# Results for 10 runs
# test_acc           = 0.4913813680410385 +- 0.0017052424639389387
# test_cross_entropy = 1.9053487062454224 +- 0.004454157142781951
# test_loss          = 1.9053486943244935 +- 0.0044541565065543305
# val_acc            = 0.5140023171901703 +- 0.0016916049277729307
# val_cross_entropy  = 1.7153084039688111 +- 0.004494663876267409
# val_loss           = 1.7153084039688111 +- 0.004494663876267409
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/ogbn-papers100m.gin mlp/config/page-rank/xl.gin
# Results for 10 runs
# test_acc           = 0.6655357539653778 +- 0.0027941891989919686
# test_cross_entropy = 1.1568999648094178 +- 0.005363392349588452
# test_loss          = 1.1568999528884887 +- 0.005363384349838102
# val_acc            = 0.6983642697334289 +- 0.0023835787384456856
# val_cross_entropy  = 0.9565011739730835 +- 0.008427189107550314
# val_loss           = 0.956501179933548 +- 0.008427197049125257
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/ogbn-papers100m.gin mlp/config/page-rank/xl.gin --bindings='epsilon=0.1'
# Results for 10 runs
# test_acc           = 0.6566147863864898 +- 0.0022851708528752093
# test_cross_entropy = 1.1794604301452636 +- 0.006059484322278056
# test_loss          = 1.1794604063034058 +- 0.006059505759989663
# val_acc            = 0.6876477837562561 +- 0.003118977373908705
# val_cross_entropy  = 0.9927952826023102 +- 0.008647591562258602
# val_loss           = 0.9927952706813812 +- 0.008647577416437922
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/ogbn-papers100m.gin mlp/config/page-rank/xl.gin --bindings='epsilon=1.0'
# Results for 10 runs
# test_acc           = 0.5154680848121643 +- 0.0014711591396321644
# test_cross_entropy = 1.823787772655487 +- 0.0036127189175831358
# test_loss          = 1.823787772655487 +- 0.0036127189175831358
# val_acc            = 0.5399353384971619 +- 0.002390429380377815
# val_cross_entropy  = 1.614618980884552 +- 0.004420480859613813
# val_loss           = 1.614618957042694 +- 0.004420476711008044


python -m graph_tf gtf_config/build_and_fit_many.gin \
    mlp/config/citeseer.gin \
    mlp/config/utils/page-rank.gin \
    --bindings='
      include_transformed_features=False
'
```

## bojchevski

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin \
      mlp/config/bojchevski/cora-full.gin
# Completed 10 trials
# test_acc           : 0.6296176910400391 +- 0.008559837216741775
# test_cross_entropy : 1.4898592114448548 +- 0.02787311236624987
# test_loss          : 0.48280423879623413 +- 0.0058826842191745865
# val_acc            : 0.6362069249153137 +- 0.006020001665530574
# val_cross_entropy  : 1.4745324015617371 +- 0.018102014099613314
# val_loss           : 0.3205566793680191 +- 0.004213327935217149
python -m graph_tf gtf_config/build_and_fit_many.gin \
      mlp/config/bojchevski/pubmed.gin
# Completed 10 trials
# test_acc           : 0.7599934160709381 +- 0.022726612494528082
# test_cross_entropy : 0.636789983510971 +- 0.06369671110012931
# test_loss          : 0.07870912030339242 +- 0.003951190571133486
# val_acc            : 0.7631665289402008 +- 0.019712504875600046
# val_cross_entropy  : 0.6424646973609924 +- 0.06468049004027754
# val_loss           : 0.5398326337337493 +- 0.05033776113824718
python -m graph_tf gtf_config/build_and_fit_many.gin \
      mlp/config/bojchevski/reddit.gin
# Completed 10 trials
# test_acc           : 0.26275125741958616 +- 0.014881027948521726
# test_cross_entropy : 2.9373538494110107 +- 0.05482792443264716
# test_loss          : 0.0735530536621809 +- 0.008829593378305054
# val_acc            : 0.26463414132595064 +- 0.013870198349030596
# val_cross_entropy  : 2.9410537004470827 +- 0.04559297693151697
# val_loss           : 0.25031524151563644 +- 0.00786853954347073
python -m graph_tf gtf_config/build_and_fit_many.gin \
      mlp/config/bojchevski/mag-coarse.gin
# Results for 10 runs
# test_acc           = 0.6162977993488312 +- 0.016971427204602776
# test_cross_entropy = 1.6589121699333191 +- 0.09144597003164721
# test_loss          = 0.008323997678235173 +- 0.0003652826421255878
# val_acc            = 0.6192499160766601 +- 0.022796306452700324
# val_cross_entropy  = 1.6649020671844483 +- 0.07138547872649152
# val_loss           = 0.5211715191602707 +- 0.022844412032875697
# NOTE: See igcn/train.py
```

## DAGNN PPR

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/cora.gin
# Completed 10 trials
# test_acc           : 0.8402999103069305 +- 0.005254509795522845
# test_cross_entropy : 0.8186442077159881 +- 0.008963442028908579
# test_loss          : 1.3812952041625977 +- 0.007245609200590164
# val_acc            : 0.8111998915672303 +- 0.005306595592325145
# val_cross_entropy  : 0.8458897531032562 +- 0.008777007530868691
# val_loss           : 1.4085407733917237 +- 0.00515481095528686
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/citeseer.gin
# Completed 10 trials
# test_acc           : 0.7286999464035034 +- 0.0058830383290848385
# test_cross_entropy : 1.2555036187171935 +- 0.005795540933335898
# test_loss          : 1.7861985921859742 +- 0.005704044232101994
# val_acc            : 0.7271999537944793 +- 0.010166605656178157
# val_cross_entropy  : 1.2711326837539674 +- 0.006089258084749853
# val_loss           : 1.8018276333808898 +- 0.005998123151973171
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/pubmed.gin
# Completed 10 trials
# test_acc           : 0.7936998963356018 +- 0.0027221251556868464
# test_cross_entropy : 0.5549510836601257 +- 0.005583555019370805
# test_loss          : 0.7556419014930725 +- 0.0057268389444282185
# val_acc            : 0.807599914073944 +- 0.004543136408530772
# val_cross_entropy  : 0.5343971014022827 +- 0.006044546636047828
# val_loss           : 0.7350879192352295 +- 0.004825770631794185
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/cs.gin
# Completed 10 trials
# test_acc           : 0.9285389840602875 +- 0.004147072214695987
# test_cross_entropy : 0.23443962931632994 +- 0.010306023018438486
# test_loss          : 0.23443961292505264 +- 0.01030601909341416
# val_acc            : 0.9157777905464173 +- 0.014244779233204895
# val_cross_entropy  : 0.275879368185997 +- 0.03923554587817029
# val_loss           : 0.275879368185997 +- 0.03923554587817029
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/physics.gin
# Completed 10 trials
# test_acc           : 0.9381713032722473 +- 0.00429702164296822
# test_cross_entropy : 0.19986981749534607 +- 0.015854435000650046
# test_loss          : 0.19986981749534607 +- 0.015854435000650046
# val_acc            : 0.9460000097751617 +- 0.018245258833181444
# val_cross_entropy  : 0.17410493046045303 +- 0.06785261678709631
# val_loss           : 0.17410492971539498 +- 0.06785261739237029
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/computer.gin
# Completed 10 trials
# test_acc           : 0.8030864059925079 +- 0.012384842044098496
# test_cross_entropy : 0.6686104953289032 +- 0.026600447010567196
# test_loss          : 0.8318828761577606 +- 0.02573517436138115
# val_acc            : 0.8656665802001953 +- 0.01932472436891266
# val_cross_entropy  : 0.42350768446922304 +- 0.053970873492456786
# val_loss           : 0.586780172586441 +- 0.05170246883844565
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/photo.gin
# Completed 10 trials
# test_acc           : 0.9070130050182342 +- 0.015791382043955568
# test_cross_entropy : 0.4129680782556534 +- 0.024699814228244618
# test_loss          : 0.8046750128269196 +- 0.023173320522718484
# val_acc            : 0.9316666722297668 +- 0.01266555804914662
# val_cross_entropy  : 0.3554733544588089 +- 0.03817515114624739
# val_loss           : 0.7471803486347198 +- 0.03874153257723484
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/ogbn-arxiv.gin
# Completed 10 trials
# test_acc           : 0.7191572666168213 +- 0.0023438061224271152
# test_cross_entropy : 0.8982750415802002 +- 0.009210794377678237
# test_loss          : 0.8982750415802002 +- 0.009210796670697064
# val_acc            : 0.7302225053310394 +- 0.0008496822451141476
# val_cross_entropy  : 0.8651066839694976 +- 0.009227173861926513
# val_loss           : 0.8651066303253174 +- 0.009227178951360116
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/ogbn-arxiv.gin --bindings='rescale_factor=0.5'
# Completed 10 trials
# test_acc           : 0.7191366970539093 +- 0.0022429145760456378
# test_cross_entropy : 0.894184809923172 +- 0.006743464074458799
# test_loss          : 0.894184821844101 +- 0.006743474262572373
# val_acc            : 0.7304440081119538 +- 0.0010909464229413564
# val_cross_entropy  : 0.8605038583278656 +- 0.006690993392476771
# val_loss           : 0.8605037927627563 +- 0.006690994906406154

```

## DAGNN Heat

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/cora.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      t=3.05
      l2_reg=2.5e-4
      dropout_rate=0.8
'
# Results for 10 runs
# test_acc           = 0.8332999050617218 +- 0.006213712205758296
# test_cross_entropy = 0.8555265486240387 +- 0.021732784518831715
# test_loss          = 1.2408257603645325 +- 0.008455156018486244
# val_acc            = 0.8117999076843262 +- 0.009734470482966747
# val_cross_entropy  = 0.8859763741493225 +- 0.021514704445964646
# val_loss           = 1.2712755799293518 +- 0.00793294695349951
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/citeseer.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      t=3.05
      l2_reg=1e-3
      dropout_rate=0.5
'
# Results for 10 runs
# test_acc           = 0.7216999471187592 +- 0.004605420169146323
# test_cross_entropy = 1.2454176187515258 +- 0.007331492761210687
# test_loss          = 1.8091169953346253 +- 0.003908255377281272
# val_acc            = 0.7279999315738678 +- 0.008485301379981974
# val_cross_entropy  = 1.2637707591056824 +- 0.006658061358543565
# val_loss           = 1.8274701356887817 +- 0.005374233422516531
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      t=3.05
      l2_reg=2.5e-4
      dropout_rate=0.8
'
# Results for 10 runs
# test_acc           = 0.7872998893260956 +- 0.0031638730580939014
# test_cross_entropy = 0.5507374048233032 +- 0.00779072919078101
# test_loss          = 0.7525762021541595 +- 0.00549256586111591
# val_acc            = 0.8049998998641967 +- 0.006082766528340938
# val_cross_entropy  = 0.527821010351181 +- 0.005846220525321949
# val_loss           = 0.7296598196029663 +- 0.0020969458784251176
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      t=5.186
      l2_reg=1.25e-4
      dropout_rate=0.8
'
# Results for 10 runs
# test_acc           : 0.7844998836517334 +- 0.0036674037642173397
# test_cross_entropy : 0.5525848925113678 +- 0.0077963919187137
# test_loss          : 0.710638576745987 +- 0.00515004325975776
# val_acc            : 0.8081998944282531 +- 0.005399992731036757
# val_cross_entropy  : 0.5225069999694825 +- 0.007619796024844372
# val_loss           : 0.6805606842041015 +- 0.0039625756734221725
```

## GCN2 PPR

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/cora.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=2.5e-4
      dropout_rate=0.7
      page_rank_dropout_rate=%dropout_rate
      force_normal=True
      verbose=False
'
```

## DGC

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/cora.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=()
      input_dropout_rate=0.
      callbacks=[]
      epochs=100
      l2_reg=1.772e-5
      t=5.27
      lr=0.156
'
# Results for 10 runs
# test_acc           = 0.8329998850822449 +- 0.0
# test_cross_entropy = 1.0332472205162049 +- 7.997179778558034e-05
# test_loss          = 1.35341637134552 +- 7.787668716583108e-05
# val_acc            = 0.791999876499176 +- 0.0
# val_cross_entropy  = 1.060354268550873 +- 9.868846063924183e-05
# val_loss           = 1.380523443222046 +- 9.77745843284022e-05
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/citeseer.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=()
      input_dropout_rate=0.
      callbacks=[]
      epochs=100
      l2_reg=1e-4
      t=3.78
      lr=1.39
'
# Results for 10 runs
# test_acc           = 0.7330999433994293 +- 0.0012999997691405334
# test_cross_entropy = 1.540098214149475 +- 0.0002505026966629614
# test_loss          = 1.7401615619659423 +- 0.0002568848600862645
# val_acc            = 0.7463999271392823 +- 0.0021540381901096243
# val_cross_entropy  = 1.549571406841278 +- 0.00032228981996688857
# val_loss           = 1.7496347665786742 +- 0.0003412241831443361
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=()
      input_dropout_rate=0.
      callbacks=[]
      epochs=100
      l2_reg=4.69e-6
      t=6.0498
      lr=0.627
'
# Results for 10 runs
# test_acc           = 0.8020999193191528 +- 0.0012999883065988858
# test_cross_entropy = 0.5606652200222015 +- 0.00029270699575537263
# test_loss          = 0.7004736661911011 +- 0.00028726485479482533
# val_acc            = 0.8099998831748962 +- 0.0
# val_cross_entropy  = 0.539818799495697 +- 0.00014758548697314097
# val_loss           = 0.679627251625061 +- 0.0002677048630478093
```

## DGC with Early Stopping

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/cora.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=()
      input_dropout_rate=0.
      l2_reg=1.772e-5
      t=5.27
      lr=0.156
'
# Results for 10 runs
# test_acc           = 0.8168999075889587 +- 0.0023000064104463287
# test_cross_entropy = 0.9663518846035004 +- 0.002694854398930253
# test_loss          = 1.3142262697219849 +- 0.0026980193981866857
# val_acc            = 0.7915999352931976 +- 0.002497970856424469
# val_cross_entropy  = 0.9944136738777161 +- 0.0029378682383645943
# val_loss           = 1.3422881007194518 +- 0.002941307690346848
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/citeseer.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=()
      input_dropout_rate=0.
      l2_reg=1e-4
      t=3.78
      lr=1.39
'
# Results for 10 runs
# test_acc           = 0.6552999377250671 +- 0.048820187375730226
# test_cross_entropy = 1.4904281616210937 +- 0.009905055921668031
# test_loss          = 2.0632964611053466 +- 0.19081971653917795
# val_acc            = 0.66619992852211 +- 0.04596911723993779
# val_cross_entropy  = 1.4943597316741943 +- 0.0090856220483037
# val_loss           = 2.0672280550003053 +- 0.19486665460917035
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=()
      input_dropout_rate=0.
      l2_reg=4.69e-6
      t=6.0498
      lr=0.627
'
# Results for 10 runs
# test_acc           = 0.7897998929023743 +- 0.004214257595276146
# test_cross_entropy = 0.5645488739013672 +- 0.005101639886952195
# test_loss          = 0.734043276309967 +- 0.02269406201857709
# val_acc            = 0.7963998913764954 +- 0.007683724008624345
# val_cross_entropy  = 0.5353948533535003 +- 0.003412482021414773
# val_loss           = 0.7048892676830292 +- 0.017289491058442397
```

With `val_acc` monitor

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/cora.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=()
      input_dropout_rate=0.
      l2_reg=1.772e-5
      t=5.27
      lr=0.156
      monitor="val_acc"
      mode="max"
'
# Results for 10 runs
# test_acc           = 0.8192998886108398 +- 0.004473275806241383
# test_cross_entropy = 1.1587880492210387 +- 0.05036158844820541
# test_loss          = 1.4687799572944642 +- 0.04485438585754552
# val_acc            = 0.8053999423980713 +- 0.0012806083599497188
# val_cross_entropy  = 1.1809003949165344 +- 0.049889655431676644
# val_loss           = 1.4908922910690308 +- 0.044403932074003276
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/citeseer.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=()
      input_dropout_rate=0.
      l2_reg=1e-4
      t=3.78
      lr=1.39
      monitor="val_acc"
      mode="max"
'
# Results for 10 runs
# test_acc           = 0.7343999147415161 +- 0.004340488352232341
# test_cross_entropy = 1.538920021057129 +- 0.0028825187230964123
# test_loss          = 1.742652451992035 +- 0.008799594376920572
# val_acc            = 0.7543999314308166 +- 0.0030724513340446113
# val_cross_entropy  = 1.5475709199905396 +- 0.003115171740485759
# val_loss           = 1.7513033509254456 +- 0.008418201800599137
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/heat.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=()
      input_dropout_rate=0.
      l2_reg=4.69e-6
      t=6.0498
      lr=0.627
      monitor="val_acc"
      mode="max"
'
# Results for 10 runs
# test_acc           = 0.7949998855590821 +- 0.004582586932897346
# test_cross_entropy = 0.569834440946579 +- 0.004747802436752615
# test_loss          = 0.7148399949073792 +- 0.020551784708387442
# val_acc            = 0.8129999577999115 +- 0.0013416190708509061
# val_cross_entropy  = 0.5464712679386139 +- 0.00233059035556487
# val_loss           = 0.6914768159389496 +- 0.016542955482656257
```

### PageRank Renormalized

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin    mlp/config/ogbn-arxiv.gin    mlp/config/utils/page-rank.gin    --bindings='
     include_transformed_features=False
     units=[1024,512]
     dropout_rate=0.5
     epsilon=[0.1]
     lr=2e-3
     renormalized=True
'
# Completed 10 trials
# test_acc           : 0.7209740340709686 +- 0.000766446683000816
# test_cross_entropy : 0.8592755436897278 +- 0.0019767978050233065
# test_loss          : 0.8592755436897278 +- 0.001976798040277611
# val_acc            : 0.7351454973220826 +- 0.0008234173377395347
# val_cross_entropy  : 0.8143201470375061 +- 0.0010195209925716463
# val_loss           : 0.8143200874328613 +- 0.0010195209925716463
python -m graph_tf gtf_config/build_and_fit_many.gin    mlp/config/ogbn-arxiv.gin    mlp/config/utils/page-rank.gin    --bindings='
     include_transformed_features=False
     units=[1024,512]
     dropout_rate=0.5
     epsilon=[0.1]
     lr=2e-3
     renormalized=False
'
# Completed 10 trials
# test_acc           : 0.7221344470977783 +- 0.0009522854482522363
# test_cross_entropy : 0.8537855744361877 +- 0.0014218224686509862
# test_loss          : 0.8537855863571167 +- 0.0014218282098619073
# val_acc            : 0.7367630124092102 +- 0.0010313576273938223
# val_cross_entropy  : 0.8094873011112214 +- 0.00113221947876514
# val_loss           : 0.8094872415065766 +- 0.00113221947876514
python -m graph_tf gtf_config/build_and_fit_many.gin    mlp/config/ogbn-arxiv.gin    mlp/config/utils/page-rank.gin    --bindings='
     include_transformed_features=False
     units=[1024,512]
     dropout_rate=0.5
     epsilon=[0.1]
     lr=2e-3
     renormalized=False
     mlp.activation="relu"
'
# Completed 10 trials
# test_acc           : 0.7208649635314941 +- 0.0007779774028526245
# test_cross_entropy : 0.8585036993026733 +- 0.0015161511505012784
# test_loss          : 0.8585036993026733 +- 0.0015161713421094234
# val_acc            : 0.7361925423145295 +- 0.0008618390264634481
# val_cross_entropy  : 0.811523151397705 +- 0.0007899324118911834
# val_loss           : 0.8115230917930603 +- 0.0007899324118911834
```

#### DAGNN PPR Hyperparam Search

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/cora.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=1.25e-4
      dropout_rate=0.8
      epsilon=0.05
'
# Results for 10 runs
# test_acc           : 0.8298999249935151 +- 0.005990837739906771
# test_cross_entropy : 0.7858397245407105 +- 0.018159031445130706
# test_loss          : 1.1035686373710631 +- 0.009594620237435077
# val_acc            : 0.8133999049663544 +- 0.006069605225679964
# val_cross_entropy  : 0.8048993110656738 +- 0.01478290852212825
# val_loss           : 1.1226282477378846 +- 0.006945821676285515
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/cora.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=2.5e-4
      dropout_rate=0.8
      epsilon=0.1
'
# Results for 10 runs
# test_acc           : 0.8414999127388001 +- 0.006576476010661232
# test_cross_entropy : 0.8898048043251038 +- 0.016688929976841092
# test_loss          : 1.2693238973617553 +- 0.009049117489767338
# val_acc            : 0.816399896144867 +- 0.007990011426375862
# val_cross_entropy  : 0.9107085883617401 +- 0.014385485546575129
# val_loss           : 1.2902276992797852 +- 0.007389739646339366
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/cora.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=5e-4
      dropout_rate=0.8
      epsilon=0.2
'
# Results for 10 runs
# test_acc           : 0.8356999099254608 +- 0.008579640232986765
# test_cross_entropy : 1.1214913725852966 +- 0.018449658725791403
# test_loss          : 1.5437161326408386 +- 0.0066263169442351616
# val_acc            : 0.8123998939990997 +- 0.006974235008831222
# val_cross_entropy  : 1.140573513507843 +- 0.01862329706645686
# val_loss           : 1.5627982854843139 +- 0.0067014513019997704

python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/citeseer.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=5e-4
      dropout_rate=0.5
      epsilon=0.05
'
# Results for 10 runs
# test_acc           : 0.7193999290466309 +- 0.005642673715190826
# test_cross_entropy : 1.0502570390701294 +- 0.0032582327120273913
# test_loss          : 1.5514662146568299 +- 0.005434077373210593
# val_acc            : 0.7181999504566192 +- 0.008784071572366046
# val_cross_entropy  : 1.0669086575508118 +- 0.002979315336919697
# val_loss           : 1.5681178450584412 +- 0.004345844013917949
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/citeseer.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=1e-3
      dropout_rate=0.5
      epsilon=0.1
'
# Results for 10 runs
# test_acc           : 0.7296999573707581 +- 0.0069720989197052
# test_cross_entropy : 1.2549948334693908 +- 0.005586586298274384
# test_loss          : 1.7843183159828186 +- 0.006895449384026319
# val_acc            : 0.7281999289989471 +- 0.009357347931402015
# val_cross_entropy  : 1.2716082692146302 +- 0.005115151669554081
# val_loss           : 1.8009317517280579 +- 0.007775077324801208
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/citeseer.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=2e-3
      dropout_rate=0.5
      epsilon=0.2
'
# Results for 10 runs
# test_acc           : 0.6739999294281006 +- 0.025569503024539468
# test_cross_entropy : 1.5719635248184205 +- 0.004325040041652383
# test_loss          : 1.8764264345169068 +- 0.005595407161609243
# val_acc            : 0.6535999357700348 +- 0.03300667018470087
# val_cross_entropy  : 1.5778702259063722 +- 0.0036224682928531716
# val_loss           : 1.8823331236839294 +- 0.006319802342547146

python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=1.25e-4
      dropout_rate=0.8
      epsilon=0.05
'
# Results for 10 runs
# test_acc           : 0.792899888753891 +- 0.004109741497597186
# test_cross_entropy : 0.5619073629379272 +- 0.013468796879218196
# test_loss          : 0.7142514765262604 +- 0.007735116994165895
# val_acc            : 0.807199913263321 +- 0.00285657465680109
# val_cross_entropy  : 0.5380625009536744 +- 0.012822794116378051
# val_loss           : 0.6904066026210784 +- 0.006664923055479865
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=2.5e-4
      dropout_rate=0.8
      epsilon=0.1
'
# Results for 10 runs
# test_acc           : 0.7949998855590821 +- 0.002898254487834139
# test_cross_entropy : 0.5554714500904083 +- 0.007830112662000377
# test_loss          : 0.7539976239204407 +- 0.004711394177188613
# val_acc            : 0.8117998957633972 +- 0.005895765352861968
# val_cross_entropy  : 0.5346618950366974 +- 0.007538318863382797
# val_loss           : 0.7331880807876587 +- 0.003445812032617452
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=5e-4
      dropout_rate=0.8
      epsilon=0.2
'
# Results for 10 runs
# test_acc           : 0.7894998908042907 +- 0.0022472194234082156
# test_cross_entropy : 0.5919919073581695 +- 0.006721776248706115
# test_loss          : 0.8349381387233734 +- 0.005110229712289505
# val_acc            : 0.8025999069213867 +- 0.006814690793088751
# val_cross_entropy  : 0.5783982574939728 +- 0.006526802632950565
# val_loss           : 0.8213444709777832 +- 0.004705365454603905


```

### Misc

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=True
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=2.5e-3
      dropout_rate=0.5
      epsilon=0.05
      unscaled=True
'
# Results for 10 runs
# test_acc           : 0.8085998892784119 +- 0.005314118868953048
# test_cross_entropy : 0.5100203394889832 +- 0.0048357435641554185
# test_loss          : 0.7451351046562195 +- 0.00815505435760153
# val_acc            : 0.8207998931407928 +- 0.00895323491807641
# val_cross_entropy  : 0.4736932575702667 +- 0.005405748711190112
# val_loss           : 0.7088080286979676 +- 0.007896139215803394
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=2.5e-3
      dropout_rate=0.5
      unscaled=True
'
# Results for 10 runs
# test_acc           : 0.8040998876094818 +- 0.006378869794097572
# test_cross_entropy : 0.5086898326873779 +- 0.0054296160988829
# test_loss          : 0.752871823310852 +- 0.008913397665114424
# val_acc            : 0.8175998747348785 +- 0.00557135419147064
# val_cross_entropy  : 0.47688226997852323 +- 0.004192240797645635
# val_loss           : 0.7210642576217652 +- 0.0061133865201747515
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/pubmed.gin mlp/config/utils/page-rank.gin --bindings='
      include_transformed_features=False
      show_progress=False
      renormalized=True
      units=(64,)
      l2_reg=2.5e-3
      dropout_rate=0.5
      unscaled=True
      epsilon=0.05
'
# Results for 10 runs
# test_acc           : 0.8049998879432678 +- 0.0031622972630813437
# test_cross_entropy : 0.518365079164505 +- 0.005032184926927742
# test_loss          : 0.7467363059520722 +- 0.005281947072463646
# val_acc            : 0.819999885559082 +- 0.007745968539255914
# val_cross_entropy  : 0.47898404896259306 +- 0.00453684102693181
# val_loss           : 0.7073552787303925 +- 0.0053696112457159335

```

### DAGNN PPR Hyperparameter Tuning

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/tune/cora.gin
# {'l2_reg': 0.00025, 'epsilon': 0.05, 'dropout_rate': 0.8}
# {'val_loss': 0.8203198313713074, 'val_cross_entropy': 0.613669216632843, 'val_acc': 0.8179998397827148, 'test_loss': 0.7845755815505981, 'test_cross_entropy': 0.5779249668121338, 'test_acc': 0.8239998817443848, 'time_this_iter_s': 2.552578926086426, 'done': True, 'timesteps_total': None, 'episodes_total': None, 'training_iteration': 10, 'trial_id': '97e0e_00013', 'experiment_id': '9bbc4810a619424a932e184f61212b41', 'date': '2022-09-30_21-55-49', 'timestamp': 1664538949, 'time_total_s': 31.900657176971436, 'pid': 11205, 'hostname': 'jackd-tuf', 'node_ip': '10.0.0.4', 'config': {'l2_reg': 0.00025, 'epsilon': 0.05, 'dropout_rate': 0.8}, 'time_since_restore': 31.900657176971436, 'timesteps_since_restore': 0, 'iterations_since_restore': 10, 'warmup_time': 0.0010006427764892578, 'experiment_tag': '13_dropout_rate=0.8000,epsilon=0.0500,l2_reg=0.0003'}
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/cora.gin --bindings='
dropout_rate=0.8
epsilon=0.05
l2_reg=0.00025
'
# Completed 10 trials
# test_acc           : 0.8188999056816101 +- 0.007725912905626729
# test_cross_entropy : 0.5745829403400421 +- 0.009643545948884903
# test_loss          : 0.8159709215164185 +- 0.02489154595645248
# val_acc            : 0.8157998979091644 +- 0.006161161954684815
# val_cross_entropy  : 0.605191832780838 +- 0.0067292505299022375
# val_loss           : 0.8465798258781433 +- 0.027386402514778706
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/cora.gin
# Completed 10 trials
# test_acc           : 0.8402999103069305 +- 0.005254509795522845
# test_cross_entropy : 0.8186442196369171 +- 0.008963443094646277
# test_loss          : 1.3812952160835266 +- 0.007245609614317786
# val_acc            : 0.8111998915672303 +- 0.005306595592325145
# val_cross_entropy  : 0.8458897531032562 +- 0.008776996509606182
# val_loss           : 1.4085407733917237 +- 0.0051548086106160676
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/tune/cora.gin --bindings='
epsilon/tune.grid_search.values = [0.1]
'
# {'l2_reg': 0.0025, 'epsilon': 0.1, 'dropout_rate': 0.5}
# {'val_loss': 1.1652953624725342, 'val_cross_entropy': 0.6741544604301453, 'val_acc': 0.8079999089241028, 'test_loss': 1.1192677021026611, 'test_cross_entropy': 0.6281267404556274, 'test_acc': 0.8359999060630798, 'time_this_iter_s': 2.898892402648926, 'done': True, 'timesteps_total': None, 'episodes_total': None, 'training_iteration': 10, 'trial_id': '95e92_00002', 'experiment_id': 'a406568358d54241ba1ee6b01b6cf76a', 'date': '2022-09-30_22-10-12', 'timestamp': 1664539812, 'time_total_s': 35.673502683639526, 'pid': 28148, 'hostname': 'jackd-tuf', 'node_ip': '10.0.0.4', 'config': {'l2_reg': 0.0025, 'epsilon': 0.1, 'dropout_rate': 0.5}, 'time_since_restore': 35.673502683639526, 'timesteps_since_restore': 0, 'iterations_since_restore': 10, 'warmup_time': 0.0014202594757080078, 'experiment_tag': '2_dropout_rate=0.5000,epsilon=0.1000,l2_reg=0.0025'}

python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/tune/citeseer.gin
# {'l2_reg': 0.01, 'epsilon': 0.1, 'dropout_rate': 0.5}
# {'val_loss': 1.895246982574463, 'val_cross_entropy': 1.2376316785812378, 'val_acc': 0.7399999499320984, 'test_loss': 1.8779215812683105, 'test_cross_entropy': 1.2203062772750854, 'test_acc': 0.7210000157356262, 'time_this_iter_s': 2.6597111225128174, 'done': True, 'timesteps_total': None, 'episodes_total': None, 'training_iteration': 10, 'trial_id': '4c9cf_00002', 'experiment_id': '3173ab3debbe417aada905e34677a111', 'date': '2022-10-01_11-30-09', 'timestamp': 1664587809, 'time_total_s': 34.57004952430725, 'pid': 28961, 'hostname': 'jackd-tuf', 'node_ip': '10.0.0.4', 'config': {'l2_reg': 0.01, 'epsilon': 0.1, 'dropout_rate': 0.5}, 'time_since_restore': 34.57004952430725, 'timesteps_since_restore': 0, 'iterations_since_restore': 10, 'warmup_time': 0.0012061595916748047, 'experiment_tag': '2_dropout_rate=0.5000,epsilon=0.1000,l2_reg=0.0100'}

python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/tune/pubmed.gin
# Best config:
# {'l2_reg': 0.0025, 'epsilon': 0.1, 'dropout_rate': 0.5}
# {'val_loss': 0.7173779010772705, 'val_cross_entropy': 0.47492119669914246, 'val_acc': 0.8139998912811279, 'test_loss': 0.7544177174568176, 'test_cross_entropy': 0.511961042881012, 'test_acc': 0.7909998893737793, 'time_this_iter_s': 2.6550915241241455, 'done': True, 'timesteps_total': None, 'episodes_total': None, 'training_iteration': 10, 'trial_id': '355e8_00008', 'experiment_id': '8eb83ed9df754241b8854738858f253a', 'date': '2022-10-01_11-53-42', 'timestamp': 1664589222, 'time_total_s': 31.434480667114258, 'pid': 12473, 'hostname': 'jackd-tuf', 'node_ip': '10.0.0.4', 'config': {'l2_reg': 0.0025, 'epsilon': 0.1, 'dropout_rate': 0.5}, 'time_since_restore': 31.434480667114258, 'timesteps_since_restore': 0, 'iterations_since_restore': 10, 'warmup_time': 0.0009343624114990234, 'experiment_tag': '8_dropout_rate=0.5000,epsilon=0.1000,l2_reg=0.0025'}

```

### SSGC Bugged

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin mlp/config/page-rank/ssgc-bugged/cora.gin
```
