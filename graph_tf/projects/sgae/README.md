# Spectral Graph Autoencoder

Unpublished work in progress. The idea is to train an `MLP` that maps spectral properties to node embeddings `Z` such that `Z @ Z^T \approx A`.

## Usage

### V1

#### spectral_size=8

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=8
dropout_rate=0.6
embedding_dim=8
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9217804431915283 +- 0.006779356608608474
# test_auc_roc = 0.9248143792152405 +- 0.006459229838819955
# test_loss    = 0.49360092282295226 +- 0.007068857351746851
# val_auc_pr   = 0.9187894999980927 +- 0.009101346778110394
# val_auc_roc  = 0.9180702447891236 +- 0.006979307370014134
# val_loss     = 0.4984939008951187 +- 0.008616022676280092
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=8
dropout_rate=0.6
embedding_dim=16
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9336817860603333 +- 0.005113321817152886
# test_auc_roc = 0.9343459844589234 +- 0.0046593877627102535
# test_loss    = 0.4868943184614182 +- 0.006362034984846824
# val_auc_pr   = 0.9330056369304657 +- 0.008309676429438865
# val_auc_roc  = 0.930025041103363 +- 0.007594630627550125
# val_loss     = 0.4913914084434509 +- 0.006483203183191374
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=8
dropout_rate=0.6
embedding_dim=32
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9408406257629395 +- 0.004754117874010575
# test_auc_roc = 0.9397019207477569 +- 0.004790813748606064
# test_loss    = 0.48300098776817324 +- 0.004958642542224261
# val_auc_pr   = 0.9370002150535583 +- 0.008243783379234378
# val_auc_roc  = 0.9331766843795777 +- 0.007422591527088267
# val_loss     = 0.48883886337280275 +- 0.006033120447892637
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=8
dropout_rate=0.6
embedding_dim=64
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.939426863193512 +- 0.0056676629862048595
# test_auc_roc = 0.9391194641590118 +- 0.005034299605942019
# test_loss    = 0.4890137821435928 +- 0.0060469123049071635
# val_auc_pr   = 0.9384441435337066 +- 0.011316627454432198
# val_auc_roc  = 0.9358469665050506 +- 0.010322018491724332
# val_loss     = 0.49194127023220063 +- 0.009027703827220387
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=8
dropout_rate=0.6
embedding_dim=128
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9412065863609314 +- 0.006299660008226889
# test_auc_roc = 0.9401416718959809 +- 0.00597898288645233
# test_loss    = 0.4865748882293701 +- 0.006254559878795776
# val_auc_pr   = 0.9373844027519226 +- 0.009539614614400064
# val_auc_roc  = 0.9339638829231263 +- 0.008084717516635238
# val_loss     = 0.49131388664245607 +- 0.00739638793584552
```

#### spectral_size=16

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=16
dropout_rate=0.6
embedding_dim=8
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9212783634662628 +- 0.007171688439175088
# test_auc_roc = 0.9269275963306427 +- 0.00614865619486642
# test_loss    = 0.49030233919620514 +- 0.007558651244980498
# val_auc_pr   = 0.9200603544712067 +- 0.00927110475860856
# val_auc_roc  = 0.921562397480011 +- 0.0064432661291395675
# val_loss     = 0.4934490829706192 +- 0.009231074719142076
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=16
dropout_rate=0.6
embedding_dim=16
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9357221007347107 +- 0.005458609832209106
# test_auc_roc = 0.9371866643428802 +- 0.005106069027998983
# test_loss    = 0.4791181296110153 +- 0.006945339768951785
# val_auc_pr   = 0.9333220064640045 +- 0.008574477837504023
# val_auc_roc  = 0.9325145661830903 +- 0.007558731555038929
# val_loss     = 0.4837421715259552 +- 0.007657315489545554
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=16
dropout_rate=0.6
embedding_dim=32
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9401239454746246 +- 0.00675798908535544
# test_auc_roc = 0.9389164030551911 +- 0.005759626885161311
# test_loss    = 0.4780155301094055 +- 0.007017795232961603
# val_auc_pr   = 0.9365861713886261 +- 0.00763814578248614
# val_auc_roc  = 0.9343882203102112 +- 0.007306328431862873
# val_loss     = 0.4836929440498352 +- 0.006929889930267803
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=16
dropout_rate=0.6
embedding_dim=64
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9406971275806427 +- 0.006501927284956296
# test_auc_roc = 0.9414263963699341 +- 0.005913907947371045
# test_loss    = 0.4829549103975296 +- 0.007536565299184084
# val_auc_pr   = 0.9396142542362214 +- 0.009007644107844036
# val_auc_roc  = 0.9382765531539917 +- 0.007296724426331099
# val_loss     = 0.486610946059227 +- 0.008129166760889144
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=16
dropout_rate=0.6
embedding_dim=128
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9420994341373443 +- 0.006079972896629702
# test_auc_roc = 0.9416660189628601 +- 0.005628609179414764
# test_loss    = 0.4779133886098862 +- 0.005865570286823184
# val_auc_pr   = 0.9394014060497284 +- 0.009039736501103587
# val_auc_roc  = 0.9377264559268952 +- 0.007061635263016391
# val_loss     = 0.4827927231788635 +- 0.008230879295813004
```

#### spectral_size=32

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=32
dropout_rate=0.6
embedding_dim=8
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9235954880714417 +- 0.008280958335069234
# test_auc_roc = 0.9280874907970429 +- 0.006247470584670922
# test_loss    = 0.4829271525144577 +- 0.00956616943207433
# val_auc_pr   = 0.9218731045722961 +- 0.010499433409712051
# val_auc_roc  = 0.9217330098152161 +- 0.009279996787090777
# val_loss     = 0.4878582715988159 +- 0.013710101632318665
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=32
dropout_rate=0.6
embedding_dim=16
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9340348422527314 +- 0.006584609285833434
# test_auc_roc = 0.9353710234165191 +- 0.005757927658588072
# test_loss    = 0.4747287958860397 +- 0.008017131889080232
# val_auc_pr   = 0.9309585273265839 +- 0.009896787948202552
# val_auc_roc  = 0.928504079580307 +- 0.007835638458720635
# val_loss     = 0.48166713714599607 +- 0.010334500787791275
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=32
dropout_rate=0.6
embedding_dim=32
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9418215572834014 +- 0.0050134860170823635
# test_auc_roc = 0.9402380406856536 +- 0.00459305833484736
# test_loss    = 0.47088381350040437 +- 0.007134110366484856
# val_auc_pr   = 0.9399993240833282 +- 0.008729079289843053
# val_auc_roc  = 0.9361744105815888 +- 0.0069799512538180775
# val_loss     = 0.47523367404937744 +- 0.010843700508579596
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=32
dropout_rate=0.6
embedding_dim=64
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9404040396213531 +- 0.0062680395225107625
# test_auc_roc = 0.9408513844013214 +- 0.004723791829111005
# test_loss    = 0.47458685636520387 +- 0.007885697769865743
# val_auc_pr   = 0.9373068392276764 +- 0.009616688256104483
# val_auc_roc  = 0.9332576930522919 +- 0.008705232625851492
# val_loss     = 0.48041634261608124 +- 0.011731589612346034
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=32
dropout_rate=0.6
embedding_dim=128
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9425512731075287 +- 0.005948791745332142
# test_auc_roc = 0.9412211358547211 +- 0.004573445989162852
# test_loss    = 0.47092358469963075 +- 0.008022772605806954
# val_auc_pr   = 0.9388704597949982 +- 0.010708413004263043
# val_auc_roc  = 0.9350995481014251 +- 0.009563835230185357
# val_loss     = 0.47871668040752413 +- 0.010958485497580607
```

#### spectral_size=64

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=64
dropout_rate=0.6
embedding_dim=8
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9202879071235657 +- 0.007029921561577445
# test_auc_roc = 0.9240114629268646 +- 0.005930449665964829
# test_loss    = 0.485672315955162 +- 0.006864382856660081
# val_auc_pr   = 0.9198180973529816 +- 0.014337318995042321
# val_auc_roc  = 0.9182184338569641 +- 0.011146510492134547
# val_loss     = 0.49097453653812406 +- 0.016488816786427586
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=64
dropout_rate=0.6
embedding_dim=16
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9337338626384735 +- 0.0062689947031205956
# test_auc_roc = 0.9334578454494477 +- 0.005387299359371263
# test_loss    = 0.4736050695180893 +- 0.008817507704784174
# val_auc_pr   = 0.9325912237167359 +- 0.013346702733279097
# val_auc_roc  = 0.9277595102787017 +- 0.01130194172663483
# val_loss     = 0.4779380440711975 +- 0.01402357590719383
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=64
dropout_rate=0.6
embedding_dim=32
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.938628900051117 +- 0.007296425607478732
# test_auc_roc = 0.9374537885189056 +- 0.004154695746176236
# test_loss    = 0.4691735327243805 +- 0.008835566136469705
# val_auc_pr   = 0.9342195451259613 +- 0.014802586424820236
# val_auc_roc  = 0.9289768815040589 +- 0.011790286647621502
# val_loss     = 0.47735168039798737 +- 0.013411567209849793
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=64
dropout_rate=0.6
embedding_dim=64
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9399124026298523 +- 0.006977462351819519
# test_auc_roc = 0.9374817252159119 +- 0.0052452743756592965
# test_loss    = 0.47124333381652833 +- 0.007964584909852667
# val_auc_pr   = 0.9356014251708984 +- 0.012464918823562128
# val_auc_roc  = 0.9299628496170044 +- 0.009454559328741775
# val_loss     = 0.4802479386329651 +- 0.0129774393784229
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=64
dropout_rate=0.6
embedding_dim=128
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9407593250274658 +- 0.0074586994262950505
# test_auc_roc = 0.9391488134860992 +- 0.005139297243213876
# test_loss    = 0.471430042386055 +- 0.00828424499369882
# val_auc_pr   = 0.9373043656349183 +- 0.013661610461784327
# val_auc_roc  = 0.9307731807231903 +- 0.01031168976111181
# val_loss     = 0.4764736771583557 +- 0.01288171533564115
```

#### spectral_size=128

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=128
dropout_rate=0.6
embedding_dim=8
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9193173348903656 +- 0.007866445914658227
# test_auc_roc = 0.9189660310745239 +- 0.007310545573047315
# test_loss    = 0.49047634899616244 +- 0.010889924829055202
# val_auc_pr   = 0.9161310970783234 +- 0.013737178694821231
# val_auc_roc  = 0.9108386695384979 +- 0.011526045716477471
# val_loss     = 0.4981973975896835 +- 0.012460898477259995
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=128
dropout_rate=0.6
embedding_dim=16
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9355837881565094 +- 0.006308116587719431
# test_auc_roc = 0.9298050105571747 +- 0.007745849531006177
# test_loss    = 0.4736542195081711 +- 0.007404842101237763
# val_auc_pr   = 0.9297020256519317 +- 0.011500391112281707
# val_auc_roc  = 0.9228158354759216 +- 0.009650657386014038
# val_loss     = 0.48426682949066163 +- 0.009994716593922029
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=128
dropout_rate=0.6
embedding_dim=32
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9408118069171906 +- 0.005489450533533613
# test_auc_roc = 0.9336446762084961 +- 0.006638855313201903
# test_loss    = 0.46996988356113434 +- 0.008153149069719461
# val_auc_pr   = 0.93388911485672 +- 0.012558538671692319
# val_auc_roc  = 0.9239854037761688 +- 0.009916416113353295
# val_loss     = 0.4806770205497742 +- 0.011973509061865997
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=128
dropout_rate=0.6
embedding_dim=64
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9395830869674683 +- 0.0058813964929363164
# test_auc_roc = 0.9310394585132599 +- 0.005442025501306111
# test_loss    = 0.47157965004444125 +- 0.008482713277666433
# val_auc_pr   = 0.9330841600894928 +- 0.011034729055902541
# val_auc_roc  = 0.9241271078586578 +- 0.00772563762012222
# val_loss     = 0.4810861200094223 +- 0.010868910946641724
python -m graph_tf gtf_config/build_and_fit_many.gin cora_v1.gin --bindings="spectral_size=128
dropout_rate=0.6
embedding_dim=128
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9443857073783875 +- 0.0043759804845399436
# test_auc_roc = 0.9379719138145447 +- 0.0046710033873973746
# test_loss    = 0.468668931722641 +- 0.006948343797505469
# val_auc_pr   = 0.9372193932533264 +- 0.013443085866434395
# val_auc_roc  = 0.9292790174484253 +- 0.011561084304267184
# val_loss     = 0.4779161632061005 +- 0.014636026031066136
```

### Citeseer

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin sgae/config/citeseer_v1.gin --bindings="spectral_size=8
dropout_rate=0.6
embedding_dim=8
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9331957399845123 +- 0.00555678991869107
# test_auc_roc = 0.932159298658371 +- 0.00441116253311428
# test_loss    = 0.4766862988471985 +- 0.006185658094956299
# val_auc_pr   = 0.9308013081550598 +- 0.009241925855734634
# val_auc_roc  = 0.9293553173542023 +- 0.008119039575664911
# val_loss     = 0.4862855911254883 +- 0.015956588717694384
python -m graph_tf gtf_config/build_and_fit_many.gin sgae/config/citeseer_v1.gin --bindings="spectral_size=16
dropout_rate=0.6
embedding_dim=16
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9458679974079132 +- 0.005095408261953487
# test_auc_roc = 0.942815500497818 +- 0.005020447453231772
# test_loss    = 0.4608936578035355 +- 0.007308562464198971
# val_auc_pr   = 0.9440015137195588 +- 0.008243401149577539
# val_auc_roc  = 0.9422946333885193 +- 0.007203025841911284
# val_loss     = 0.4694440960884094 +- 0.014485388626232127
python -m graph_tf gtf_config/build_and_fit_many.gin sgae/config/citeseer_v1.gin --bindings="spectral_size=32
dropout_rate=0.6
embedding_dim=32
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9542889654636383 +- 0.004255988524177745
# test_auc_roc = 0.9505551040172577 +- 0.005284599035955211
# test_loss    = 0.45162923634052277 +- 0.007175754034078689
# val_auc_pr   = 0.9515808284282684 +- 0.00717147511951795
# val_auc_roc  = 0.9478710889816284 +- 0.006488893952566154
# val_loss     = 0.4588778555393219 +- 0.013543768871573802
python -m graph_tf gtf_config/build_and_fit_many.gin sgae/config/citeseer_v1.gin --bindings="spectral_size=64
dropout_rate=0.6
embedding_dim=64
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.954550278186798 +- 0.0055728708544012665
# test_auc_roc = 0.9491417586803437 +- 0.005440700679982893
# test_loss    = 0.4542819619178772 +- 0.008523243339674351
# val_auc_pr   = 0.9528714120388031 +- 0.011316355894826085
# val_auc_roc  = 0.9472200512886048 +- 0.01224086097013974
# val_loss     = 0.46178652346134186 +- 0.018934663329453268
python -m graph_tf gtf_config/build_and_fit_many.gin sgae/config/citeseer_v1.gin --bindings="spectral_size=64
dropout_rate=0.6
embedding_dim=64
hidden_units=256
"
# Results for 10 runs
# test_auc_pr  = 0.9498191058635712 +- 0.00689810353375469
# test_auc_roc = 0.9449729800224305 +- 0.004425194571830849
# test_loss    = 0.4662816435098648 +- 0.011772091423487768
# val_auc_pr   = 0.9530498743057251 +- 0.008388892923423092
# val_auc_roc  = 0.9477188348770141 +- 0.008291369045697515
# val_loss     = 0.46562792360782623 +- 0.011169169656551325
```

### Tune

```bash
python -m graph_tf projects/sgae/config/tune/v1/cora.gin

# Hyperparameter    |Value             |Best Value So Far
# spectral_size     |8                 |8
# dropout_rate      |0.3               |0.3
# embedding_size    |64                |64
# hidden_units      |128               |128
# hidden_layers     |1                 |1
# use_laplacian     |True              |True
```
