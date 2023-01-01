# Deep Adaptive Graph Neural Networks

- [Original paper](https://arxiv.org/abs/2007.09296)
- [Repository](https://github.com/divelab/DeeperGNN)

```bibtex
@inproceedings{liu2020towards,
  title={Towards deeper graph neural networks},
  author={Liu, Meng and Gao, Hongyang and Ji, Shuiwang},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={338--348},
  year={2020}
}
```

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/cora.gin
# Completed 10 trials
# test_acc           : 0.8455000162124634 +- 0.005315082416125652
# test_cross_entropy : 0.5826616227626801 +- 0.004559773324750922
# test_loss          : 1.183704662322998 +- 0.008627757058138332
# val_acc            : 0.8225999057292939 +- 0.003799987780565693
# val_cross_entropy  : 0.6268380999565124 +- 0.004791394830097534
# val_loss           : 1.2278811931610107 +- 0.007549563756895697
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/citeseer.gin
# Completed 10 trials
# test_acc           : 0.7298999905586243 +- 0.004721226575200714
# test_cross_entropy : 1.1339227557182312 +- 0.005171815413990156
# test_loss          : 1.8822817921638488 +- 0.006716728565565693
# val_acc            : 0.7318000078201294 +- 0.005399992730905175
# val_cross_entropy  : 1.1574506521224976 +- 0.006250679831598071
# val_loss           : 1.9058096408843994 +- 0.004636005213560762
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/pubmed.gin
# Completed 10 trials
# test_acc           : 0.8052999138832092 +- 0.0038999964030446877
# test_cross_entropy : 0.5286964356899262 +- 0.011758856105612165
# test_loss          : 0.7566060841083526 +- 0.013895947223560188
# val_acc            : 0.8200000286102295 +- 0.008390476423019579
# val_cross_entropy  : 0.4865698337554932 +- 0.006490690376447124
# val_loss           : 0.7144794285297393 +- 0.006709896785750308




python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/cs.gin
# Completed 10 trials
# test_acc           : 0.9303020417690278 +- 0.005413072519873217
# test_cross_entropy : 0.22692469954490663 +- 0.016255541366450597
# test_loss          : 0.2269246906042099 +- 0.016255547135734398
# val_acc            : 0.9171110987663269 +- 0.00999012504609339
# val_cross_entropy  : 0.2618979141116142 +- 0.035676951847438375
# val_loss           : 0.26189791560173037 +- 0.03567695104217406
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/physics.gin
# Completed 10 trials
# test_acc           : 0.9349764883518219 +- 0.00852504167057135
# test_cross_entropy : 0.22231295108795165 +- 0.035442895081068584
# test_loss          : 0.22231294959783554 +- 0.03544289416305682
# val_acc            : 0.9493333399295807 +- 0.020483060625070475
# val_cross_entropy  : 0.17933755367994308 +- 0.0729532016550183
# val_loss           : 0.17933755367994308 +- 0.0729532016550183
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/computer.gin
# Completed 10 trials
# test_acc           : 0.8327924907207489 +- 0.01437127294141898
# test_cross_entropy : 0.5872298181056976 +- 0.039609049470652524
# test_loss          : 0.7113358616828919 +- 0.04111944747895341
# val_acc            : 0.8866666734218598 +- 0.022360699322747833
# val_cross_entropy  : 0.37639028429985044 +- 0.06366644785409582
# val_loss           : 0.5004963189363479 +- 0.0695039998479898
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/photo.gin
# Completed 10 trials
# test_acc           : 0.913517701625824 +- 0.013190623487733283
# test_cross_entropy : 0.36807321310043334 +- 0.016032499267160282
# test_loss          : 0.6996791899204254 +- 0.014389251657444008
# val_acc            : 0.9408333837985993 +- 0.011456466898590256
# val_cross_entropy  : 0.3082620620727539 +- 0.03511848117248903
# val_loss           : 0.6398680508136749 +- 0.035678173252923266
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/ogbn-arxiv.gin
# Completed 10 trials
# test_acc           : 0.7193918108940125 +- 0.0021259328851754387
# test_cross_entropy : 0.9093454062938691 +- 0.00539957296144633
# test_loss          : 0.909345418214798 +- 0.005399568838733989
# val_acc            : 0.7300110995769501 +- 0.0011696621805763687
# val_cross_entropy  : 0.8738760113716125 +- 0.00342012461572797
# val_loss           : 0.8738759517669678 +- 0.00342012883208192
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/ogbn-arxiv.gin --bindings='simplified=True'
# Completed 10 trials
# test_acc           : 0.7139703452587127 +- 0.0022895158428986977
# test_cross_entropy : 0.9408879280090332 +- 0.00493484961739268
# test_loss          : 0.9408879339694977 +- 0.0049348550654423435
# val_acc            : 0.7244370996952056 +- 0.0008584899620374919
# val_cross_entropy  : 0.9058721005916596 +- 0.0026121406156573083
# val_loss           : 0.9058720290660858 +- 0.0026121302849778164
```

### Hyperparameter Tuning

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/tune/cora.gin
# Best config:
# {'l2_reg': 0.0025, 'num_propagations': 10, 'dropout_rate': 0.8}
# {'val_loss': 1.2178921699523926, 'val_cross_entropy': 0.6225194931030273, 'val_acc': 0.8299998641014099, 'test_loss': 1.1707351207733154, 'test_cross_entropy': 0.5753625631332397, 'test_acc': 0.8429999947547913, 'time_this_iter_s': 8.33519959449768, 'done': True, 'timesteps_total': None, 'episodes_total': None, 'training_iteration': 10, 'trial_id': '0cbc9_00011', 'experiment_id': 'e5a5b349a94643d38471d4274f984146', 'date': '2022-09-30_21-04-11', 'timestamp': 1664535851, 'time_total_s': 69.79944705963135, 'pid': 11611, 'hostname': 'jackd-tuf', 'node_ip': '10.0.0.4', 'config': {'l2_reg': 0.0025, 'num_propagations': 10, 'dropout_rate': 0.8}, 'time_since_restore': 69.79944705963135, 'timesteps_since_restore': 0, 'iterations_since_restore': 10, 'warmup_time': 0.0012884140014648438, 'experiment_tag': '11_dropout_rate=0.8000,l2_reg=0.0025,num_propagations=10'}
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/tune/cora.gin --bindings='
tune_metric = "val_cross_entropy"
tune_mode = "min"
'
# Best config:
# {'l2_reg': 0.00025, 'num_propagations': 20, 'dropout_rate': 0.8}
# {'val_loss': 0.7247505187988281, 'val_cross_entropy': 0.5731000900268555, 'val_acc': 0.807999849319458, 'test_loss': 0.6714138388633728, 'test_cross_entropy': 0.5197634696960449, 'test_acc': 0.8299999833106995, 'time_this_iter_s': 3.7801647186279297, 'done': True, 'timesteps_total': None, 'episodes_total': None, 'training_iteration': 10, 'trial_id': 'b61ac_00021', 'experiment_id': '91c9754290b04d96b4182c589ebff479', 'date': '2022-09-30_21-39-33', 'timestamp': 1664537973, 'time_total_s': 39.56362271308899, 'pid': 26602, 'hostname': 'jackd-tuf', 'node_ip': '10.0.0.4', 'config': {'l2_reg': 0.00025, 'num_propagations': 20, 'dropout_rate': 0.8}, 'time_since_restore': 39.56362271308899, 'timesteps_since_restore': 0, 'iterations_since_restore': 10, 'warmup_time': 0.0009872913360595703, 'experiment_tag': '21_dropout_rate=0.8000,l2_reg=0.0003,num_propagations=20'}
```
