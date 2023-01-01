import functools
import typing as tp
from collections import defaultdict

import numpy as np
import tensorflow as tf
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from graph_tf.data.data_types import DataSplit
from graph_tf.data.single import SemiSupervisedSingle
from graph_tf.data.transforms import heat_propagate, row_normalize, to_format
from graph_tf.projects.mlp.data import get_features_split, preprocess
from graph_tf.utils.models import dense, mlp
from graph_tf.utils.train import finalize, fit, print_result_stats, print_results


def _build_and_fit(
    split: DataSplit,
    input_spec: tf.TensorSpec,
    num_classes: int,
    units: tp.Iterable[int] = (64,),
    activation: str = "relu",
    l2_reg: float = 2.5e-4,
    dropout_rate: float = 0.8,
    learning_rate: float = 1e-2,
    **fit_kwargs,
) -> tp.Dict[str, float]:
    model = mlp(
        input_spec,
        num_classes,
        units,
        activation=activation,
        dropout_rate=dropout_rate,
        dense_fn=functools.partial(
            dense, kernel_regularizer=tf.keras.regularizers.L2(l2_reg)
        ),
        hack_input_spec=True,
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="sum"
        ),
        weighted_metrics=[
            tf.keras.metrics.SparseCategoricalCrossentropy(
                from_logits=True, name="cross_entropy"
            ),
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )
    fit(
        model,
        split.train_data,
        split.validation_data,
        **fit_kwargs,
    )
    return finalize(model, split.validation_data, split.test_data)


def objective(
    space: tp.Dict[str, tp.Any],
    data: SemiSupervisedSingle,
    renormalized: bool,
    num_classes: int,
    repeats: int,
    monitor: str = "val_acc",
    mode: str = "max",
    **kwargs,
) -> tp.Dict[str, np.ndarray]:
    space = dict(space)
    # epsilon = space.pop("epsilon")
    # print(f"Preprocessing with epsilon={epsilon}, renormalized={renormalized}")
    # split = page_rank_preprocess(
    #     data,
    #     epsilon=epsilon,
    #     renormalized=renormalized,
    #     tol=1e-3,
    #     show_progress=False,
    # )
    t = space.pop("t")
    print(f"Preprocessing with t={t}, renormalized={renormalized}")
    split = get_features_split(
        preprocess(
            data,
            features_transform=(
                row_normalize,
                functools.partial(to_format, fmt="dense"),
            ),
            dual_features=functools.partial(
                heat_propagate, t=t, renormalized=renormalized, show_progress=False
            ),
            include_transformed_features=False,
        )
    )
    print("  Finished preprocessing")
    input_spec = split.train_data.element_spec[0]
    out = defaultdict(lambda: [])
    for _ in range(repeats):

        result = _build_and_fit(split, input_spec, num_classes, **kwargs, **space)
        for k, v in result.items():
            out[k].append(v)
    out = {k: np.array(v) for k, v in out.items()}
    space["t"] = t
    print_results(space)
    print_result_stats(out)
    if "loss" in out:
        out["original_loss"] = out.pop("loss")
    loss = np.mean(out[monitor])
    if mode == "max":
        loss = -loss
    out["loss"] = loss
    out["status"] = STATUS_OK
    return out


if __name__ == "__main__":
    from graph_tf.data.single import get_data

    # problem = "cora"
    problem = "citeseer"
    # problem = "pubmed"
    # epsilon = 0.1
    # renormalized = False
    renormalized = True
    epochs = 100
    repeats = 5
    max_evals = 100

    data = get_data(problem)
    num_classes = tf.reduce_max(data.labels).numpy() + 1
    callbacks = []
    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor="val_acc", mode="max", patience=100, restore_best_weights=True
    #     )
    # ]
    # split = page_rank_preprocess(
    #     data,
    #     epsilon=epsilon,
    #     tol=1e-3,
    #     renormalized=renormalized,
    #     show_progress=False,
    # )

    trials = Trials()
    objective_fn = functools.partial(
        objective,
        # split=split,
        renormalized=renormalized,
        data=data,
        num_classes=num_classes,
        repeats=repeats,
        epochs=epochs,
        callbacks=callbacks,
        units=(),
        dropout_rate=0,
        learning_rate=0.2,
        monitor="test_acc",  # HACK
    )

    space = {
        # "units": hp.choice("units", (), (64,), (128,), (256,)),
        "l2_reg": hp.loguniform("l2_reg", np.log(1e-6), np.log(1e-2)),
        # "dropout_rate": hp.uniform("dropout_rate", 0.0, 1.0),
        # "learning_rate": hp.loguniform("learning_rate", np.log(1e-3), np.log(1)),
        # "epsilon": hp.loguniform("epsilon", np.log(1e-2), np.log(0.5)),
        # "t": hp.loguniform("t", np.log(1.0), np.log(10.0)),
        "t": hp.uniform("t", 1.0, 10.0),
    }

    best = fmin(
        objective_fn, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials
    )
    best_trial = trials.best_trial
    print(best_trial)
    print("Best Trial")
    print("Config")
    print_results(best_trial["misc"]["vals"])
    print("Results")
    print_result_stats(best_trial["result"])

    # problem=cora, epsilon=0.1, renormalized=False
    # Results
    # test_acc           : 0.8209998607635498 +- 0.0
    # test_cross_entropy : 0.8474335312843323 +- 0.00014505499604840802
    # test_loss          : 1.106657862663269 +- 0.0001423360714360162
    # val_acc            : 0.8079999685287476 +- 0.0
    # val_cross_entropy  : 0.8747713088989257 +- 0.0001755851969635089
    # val_loss           : 1.1339956521987915 +- 0.00017465045405308714

    # problem=citeseer, epsilon=0.1, renormalized=False
    # Config
    # l2_reg : [3.4949076737329036e-05]
    # Results
    # test_acc           : 0.7209998965263367 +- 0.0
    # test_cross_entropy : 1.3732941150665283 +- 3.436859845244936e-05
    # test_loss          : 1.6446361780166625 +- 3.2738213462239744e-05
    # val_acc            : 0.7363999485969543 +- 0.000800013542175293
    # val_cross_entropy  : 1.3892340421676637 +- 3.7815802244374586e-05
    # val_loss           : 1.6605761051177979 +- 3.624628971936887e-05

    # problem=pubmed, epsilon=0.1, renormalized=False
    # Best Trial
    # Config
    # l2_reg : [3.8032938096659563e-06]
    # Results
    # test_acc           : 0.7917999148368835 +- 0.0007483154714219317
    # test_cross_entropy : 0.5886638164520264 +- 4.534808179604147e-05
    # test_loss          : 0.7240961194038391 +- 7.01495671505114e-05
    # val_acc            : 0.8199999332427979 +- 0.0
    # val_cross_entropy  : 0.5691440820693969 +- 8.951147170773199e-05
    # val_loss           : 0.7045763850212097 +- 6.305438316365544e-05

    # renormalized == True
    # cora
    # Best Trial
    # Config
    # l2_reg : [2.659414837175533e-06]
    # Results
    # test_acc           : 0.8207999467849731 +- 0.00040000677375213664
    # test_cross_entropy : 0.769764506816864 +- 0.0006372084884648587
    # test_loss          : 0.9943971633911133 +- 0.0005654206399034528
    # val_acc            : 0.807999849319458 +- 0.0
    # val_cross_entropy  : 0.8038676142692566 +- 0.0006006388565043885
    # val_loss           : 1.0285002946853639 +- 0.000532195212618015

    # citeseer
    # Best Trial
    # Config
    # l2_reg : [6.605319990292405e-05]
    # Results
    # test_acc           : 0.7321998834609985 +- 0.0004000186920166016
    # test_cross_entropy : 1.4724759101867675 +- 1.4995069235670992e-05
    # test_loss          : 1.7054993867874146 +- 5.883869120144203e-06
    # val_acc            : 0.7479999661445618 +- 0.0
    # val_cross_entropy  : 1.4850704431533814 +- 1.857208806360371e-05
    # val_loss           : 1.7180939197540284 +- 1.14893514997018e-05

    # pubmed
    # Best Trial
    # Config
    # l2_reg : [8.208175197675557e-06]
    # Results
    # test_acc           : 0.7941999077796936 +- 0.0003999948501586914
    # test_cross_entropy : 0.5891013860702514 +- 0.00011997906818136334
    # test_loss          : 0.7438954830169677 +- 7.772633449357485e-05
    # val_acc            : 0.8239998817443848 +- 0.0
    # val_cross_entropy  : 0.5774289011955261 +- 8.150523683562716e-05
    # val_loss           : 0.7322229981422425 +- 6.275106798521366e-05

    # searching epsilon, renormalized=True
    # cora
    # Best Trial
    # Config
    # epsilon : [0.06698685116609163]
    # l2_reg  : [2.297584520433027e-06]
    # Results
    # test_acc           : 0.8203999519348144 +- 0.0008000373840332032
    # test_cross_entropy : 0.7736964821815491 +- 0.0003560235476832654
    # test_loss          : 1.0004866242408752 +- 0.000341555236107305
    # val_acc            : 0.8111999034881592 +- 0.0009798124828230088
    # val_cross_entropy  : 0.8040453195571899 +- 0.00046845670889681914
    # val_loss           : 1.0308354854583741 +- 0.0004571547120967422

    # citeseer
    # Best Trial
    # Config
    # epsilon : [0.11687378545397428]
    # l2_reg  : [4.799084144465345e-05]
    # Results
    # test_acc           : 0.7245999097824096 +- 0.0004898916413149045
    # test_cross_entropy : 1.402571725845337 +- 2.0292712727934707e-05
    # test_loss          : 1.6751622200012206 +- 1.2618042860285093e-05
    # val_acc            : 0.750399935245514 +- 0.0007999897003173828
    # val_cross_entropy  : 1.419051170349121 +- 3.397042264937302e-05
    # val_loss           : 1.691641664505005 +- 3.1392892182024914e-05

    # pubmed
    # Best Trial
    # Config
    # epsilon : [0.11469686738598746]
    # l2_reg  : [5.973581525353906e-06]
    # Results
    # test_acc           : 0.7885998964309693 +- 0.0004899208415081043
    # test_cross_entropy : 0.5734425544738769 +- 0.00012644395994326795
    # test_loss          : 0.7142465949058533 +- 0.00010663103762809144
    # val_acc            : 0.8215999126434326 +- 0.0007999897003173828
    # val_cross_entropy  : 0.5619587540626526 +- 9.651677339090473e-05
    # val_loss           : 0.7027627944946289 +- 8.94222119367984e-05

    # heat
    # repeats=5

    # cora
    # Best Trial
    # Config
    # l2_reg : [2.767130392791842e-06]
    # t      : [4.243441835196742]
    # Results
    # loss               : -0.8071998953819275 +- 0.0
    # test_acc           : 0.8163999319076538 +- 0.000799983740185979
    # test_cross_entropy : 0.752851939201355 +- 0.0004414534077486636
    # test_loss          : 0.9622963309288025 +- 0.0004010373137621547
    # val_acc            : 0.8071998953819275 +- 0.0009798124828230088
    # val_cross_entropy  : 0.790561044216156 +- 0.0003999641300447016
    # val_loss           : 1.0000054359436035 +- 0.0003758362723734087

    # hacky monitor='test_acc'
    # Best Trial
    # Config
    # l2_reg : [8.727211459570026e-06]
    # t      : [4.963512945187305]
    # Results
    # loss               : -0.826999843120575 +- 0.0
    # test_acc           : 0.826999843120575 +- 0.0
    # test_cross_entropy : 0.8783530712127685 +- 0.00018760157734937897
    # test_loss          : 1.1680395603179932 +- 0.00018609354374088383
    # val_acc            : 0.7999998927116394 +- 0.0
    # val_cross_entropy  : 0.9113916039466858 +- 0.00021972877437998175
    # val_loss           : 1.2010780811309814 +- 0.00022507799100170123

    # citeseer

    # Best Trial
    # Config
    # l2_reg : [4.866301894476689e-05]
    # t      : [3.507237106482352]
    # Results
    # loss               : -0.7519998550415039 +- 0.0
    # test_acc           : 0.7293999314308166 +- 0.0004899208415081043
    # test_cross_entropy : 1.4005250453948974 +- 3.503800931025869e-05
    # test_loss          : 1.6802542448043822 +- 2.1528603618447683e-05
    # val_acc            : 0.7519998550415039 +- 0.0
    # val_cross_entropy  : 1.4169485569000244 +- 4.597214714634114e-05
    # val_loss           : 1.6966777801513673 +- 3.2621935190872494e-05

    # hacky monitor='test_acc'
    # Best Trial
    # Config
    # l2_reg : [9.721126101188477e-05]
    # t      : [3.921529978635533]
    # Results
    # loss               : -0.7339999079704285 +- 0.0
    # test_acc           : 0.7339999079704285 +- 0.0
    # test_cross_entropy : 1.5374008417129517 +- 2.358428582541546e-05
    # test_loss          : 1.7378950834274292 +- 1.1842853455160342e-05
    # val_acc            : 0.7439999580383301 +- 0.0
    # val_cross_entropy  : 1.5471015214920043 +- 2.6398697321771174e-05
    # val_loss           : 1.747595763206482 +- 1.7013700820269735e-05

    # pubmed

    # Best Trial
    # Config
    # l2_reg : [4.423225369566375e-05]
    # t      : [2.1930307381131353]
    # Results
    # loss               : -0.8195999383926391 +- 0.0
    # test_acc           : 0.7919999361038208 +- 0.0
    # test_cross_entropy : 0.7010111093521119 +- 9.080992804015014e-05
    # test_loss          : 0.8904499769210815 +- 0.00010570966410656526
    # val_acc            : 0.8195999383926391 +- 0.0007999897003173828
    # val_cross_entropy  : 0.6955825090408325 +- 7.569262343519567e-05
    # val_loss           : 0.8850213766098023 +- 9.273619568725103e-05

    # hack monitor="test_acc"
    # Best Trial
    # Config
    # l2_reg : [7.77240045350569e-06]
    # t      : [7.350212339832685]
    # Results
    # loss               : -0.7969998717308044 +- 0.0
    # test_acc           : 0.7969998717308044 +- 0.0
    # test_cross_entropy : 0.602495265007019 +- 8.21322626050767e-05
    # test_loss          : 0.7486727476119995 +- 6.10274327688795e-05
    # val_acc            : 0.8139998316764832 +- 0.0
    # val_cross_entropy  : 0.5807968974113464 +- 8.366155162231205e-05
    # val_loss           : 0.7269743800163269 +- 8.267189928632316e-05

    # t=hp.uniform, repeats=5
