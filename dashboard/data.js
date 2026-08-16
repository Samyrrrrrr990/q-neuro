window.QNEURO_DATA = {
  "candidates": [
    {
      "ambiguity_nll": 1.7506857713063557,
      "candidate_id": "adaptive_attractor",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "attractor",
        "measurement": "distance_energy",
        "state": "real",
        "training": "adamw",
        "transition": "energy_descent_soft_act"
      },
      "in_domain_top1": 0.7239999969800314,
      "parameter_count": 19801.0,
      "parents": [
        "energy_attractor"
      ],
      "pareto": true,
      "shifted_ece": 0.32969709237416583,
      "shifted_nll": 2.5179518858591714,
      "shifted_top1": 0.4306666685475244,
      "source_experiment": "QN-000014",
      "training_seconds": 1.263418971997453
    },
    {
      "ambiguity_nll": 2.106645663579305,
      "candidate_id": "complex_mlp",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "feedforward",
        "measurement": "born",
        "state": "complex",
        "training": "adamw",
        "transition": "none"
      },
      "in_domain_top1": 0.7076666553815206,
      "parameter_count": 19982.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.10622759742869271,
      "shifted_nll": 2.034939103656345,
      "shifted_top1": 0.3997777799765269,
      "source_experiment": "QN-000014",
      "training_seconds": 0.25511108366966556
    },
    {
      "ambiguity_nll": 2.3517801761627197,
      "candidate_id": "complex_operator",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 1.0,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "adamw",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.969000001748403,
      "parameter_count": 20304.0,
      "parents": [
        "real_operator"
      ],
      "pareto": true,
      "shifted_ece": 0.25403943326738143,
      "shifted_nll": 1.5149859587351482,
      "shifted_top1": 0.6471111112170749,
      "source_experiment": "QN-000014",
      "training_seconds": 4.5339992916681995
    },
    {
      "ambiguity_nll": 1.5297951300938923,
      "candidate_id": "coupled_tensor",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "factorized",
        "measurement": "affine",
        "state": "real",
        "training": "adamw",
        "transition": "multiplicative"
      },
      "in_domain_top1": 0.7193333307902018,
      "parameter_count": 19961.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.13571971737676194,
      "shifted_nll": 1.9623925685882568,
      "shifted_top1": 0.38333333200878567,
      "source_experiment": "QN-000014",
      "training_seconds": 0.2483690139997634
    },
    {
      "ambiguity_nll": 1.4683514038721721,
      "candidate_id": "density_dynamics",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.6350000103314718,
      "generation": 0,
      "genome": {
        "family": "density",
        "measurement": "diagonal",
        "state": "density_complex_rank2",
        "training": "adamw",
        "transition": "hamiltonian_dissipative"
      },
      "in_domain_top1": 0.890666663646698,
      "parameter_count": 16240.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.22634234693315292,
      "shifted_nll": 2.0448915561040244,
      "shifted_top1": 0.4525555570920308,
      "source_experiment": "QN-000014",
      "training_seconds": 5.122936624999663
    },
    {
      "ambiguity_nll": 1.8940201997756958,
      "candidate_id": "dissipative",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.008333333147068819,
      "generation": 0,
      "genome": {
        "family": "dynamical",
        "measurement": "born",
        "state": "complex",
        "training": "adamw",
        "transition": "dissipative"
      },
      "in_domain_top1": 0.7226666808128357,
      "parameter_count": 20020.0,
      "parents": [
        "hamiltonian"
      ],
      "pareto": false,
      "shifted_ece": 0.13576079242759279,
      "shifted_nll": 1.8471499813927543,
      "shifted_top1": 0.43766666783226865,
      "source_experiment": "QN-000014",
      "training_seconds": 3.9896435280000637
    },
    {
      "ambiguity_nll": 1.7285333077112834,
      "candidate_id": "energy_attractor",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "attractor",
        "measurement": "distance_energy",
        "state": "real",
        "training": "adamw",
        "transition": "energy_descent"
      },
      "in_domain_top1": 0.7169999877611796,
      "parameter_count": 19797.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.3109850254323747,
      "shifted_nll": 2.5493748452928333,
      "shifted_top1": 0.4201111098130544,
      "source_experiment": "QN-000014",
      "training_seconds": 0.7416567640054078
    },
    {
      "ambiguity_nll": 2.304250717163086,
      "candidate_id": "graph_network",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "graph",
        "measurement": "affine",
        "state": "real_nodes",
        "training": "adamw",
        "transition": "message_passing"
      },
      "in_domain_top1": 0.3193333347638448,
      "parameter_count": 19676.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.06265296869807774,
      "shifted_nll": 2.4206685754987927,
      "shifted_top1": 0.18366666634877524,
      "source_experiment": "QN-000014",
      "training_seconds": 1.7931153190002078
    },
    {
      "ambiguity_nll": 2.295586188634237,
      "candidate_id": "gru",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.996666669845581,
      "generation": 0,
      "genome": {
        "family": "recurrent",
        "measurement": "affine",
        "state": "real",
        "training": "adamw",
        "transition": "gated_recurrence"
      },
      "in_domain_top1": 0.987333337465922,
      "parameter_count": 19656.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.3995928532547421,
      "shifted_nll": 3.848475615183512,
      "shifted_top1": 0.24744444092114767,
      "source_experiment": "QN-000014",
      "training_seconds": 2.5763887359983833
    },
    {
      "ambiguity_nll": 1.870278795560201,
      "candidate_id": "hamiltonian",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.9833333293596903,
      "generation": 0,
      "genome": {
        "family": "dynamical",
        "measurement": "born",
        "state": "complex",
        "training": "adamw",
        "transition": "hamiltonian"
      },
      "in_domain_top1": 0.9640000065167745,
      "parameter_count": 19998.0,
      "parents": [
        "complex_operator"
      ],
      "pareto": false,
      "shifted_ece": 0.27321215801768833,
      "shifted_nll": 1.7654191785388524,
      "shifted_top1": 0.5562222202618917,
      "source_experiment": "QN-000014",
      "training_seconds": 4.646453749999637
    },
    {
      "ambiguity_nll": 1.4459930658340454,
      "candidate_id": "hopfield",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "associative",
        "measurement": "similarity",
        "state": "real",
        "training": "adamw",
        "transition": "retrieval"
      },
      "in_domain_top1": 0.6023333271344503,
      "parameter_count": 19816.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.04514752421528101,
      "shifted_nll": 1.8774058024088542,
      "shifted_top1": 0.35355555680063033,
      "source_experiment": "QN-000014",
      "training_seconds": 0.8330520556652724
    },
    {
      "ambiguity_nll": 1.8671486377716064,
      "candidate_id": "hybrid_dynamics",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.9233333269755045,
      "generation": 0,
      "genome": {
        "family": "dynamical",
        "measurement": "born",
        "state": "complex",
        "training": "adamw",
        "transition": "hamiltonian_dissipative"
      },
      "in_domain_top1": 0.9316666722297668,
      "parameter_count": 19894.0,
      "parents": [
        "hamiltonian"
      ],
      "pareto": false,
      "shifted_ece": 0.27466307414902585,
      "shifted_nll": 1.7710943089591131,
      "shifted_top1": 0.550111108356052,
      "source_experiment": "QN-000014",
      "training_seconds": 4.948009250001633
    },
    {
      "ambiguity_nll": 1.4449293216069539,
      "candidate_id": "logistic",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "linear",
        "measurement": "affine",
        "state": "none",
        "training": "adamw",
        "transition": "none"
      },
      "in_domain_top1": 0.7249999841054281,
      "parameter_count": 1660.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.15315402381949952,
      "shifted_nll": 2.269262181388007,
      "shifted_top1": 0.35166666573948335,
      "source_experiment": "QN-000014",
      "training_seconds": 0.11713226400024723
    },
    {
      "ambiguity_nll": 1.532847801844279,
      "candidate_id": "mlp",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "feedforward",
        "measurement": "affine",
        "state": "real",
        "training": "adamw",
        "transition": "none"
      },
      "in_domain_top1": 0.7263333201408386,
      "parameter_count": 20002.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.1653942863146464,
      "shifted_nll": 2.0838643312454224,
      "shifted_top1": 0.379111111164093,
      "source_experiment": "QN-000014",
      "training_seconds": 0.1778495413333682
    },
    {
      "ambiguity_nll": 1.4182816346486409,
      "candidate_id": "real_operator",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.9983333349227905,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "affine",
        "state": "real",
        "training": "adamw",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.9763333400090536,
      "parameter_count": 19901.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.10478101298213005,
      "shifted_nll": 1.8178126414616902,
      "shifted_top1": 0.4947777754730649,
      "source_experiment": "QN-000014",
      "training_seconds": 2.9089960970037887
    },
    {
      "ambiguity_nll": 2.076831817626953,
      "candidate_id": "state_space",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 1.0,
      "generation": 0,
      "genome": {
        "family": "state_space",
        "measurement": "affine",
        "state": "real",
        "training": "adamw",
        "transition": "diagonal_recurrence"
      },
      "in_domain_top1": 0.9776666561762491,
      "parameter_count": 19970.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.3759559094905853,
      "shifted_nll": 3.2191674974229603,
      "shifted_top1": 0.33744444118605715,
      "source_experiment": "QN-000014",
      "training_seconds": 1.634957138999501
    },
    {
      "ambiguity_nll": 4.344689051310222,
      "candidate_id": "transformer",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.8433333237965902,
      "generation": 0,
      "genome": {
        "family": "attention",
        "measurement": "affine",
        "state": "real_sequence",
        "training": "adamw",
        "transition": "self_attention"
      },
      "in_domain_top1": 0.8666666547457377,
      "parameter_count": 18980.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.18292923602792951,
      "shifted_nll": 2.149098051918877,
      "shifted_top1": 0.4934444394376543,
      "source_experiment": "QN-000014",
      "training_seconds": 2.060012611669663
    },
    {
      "ambiguity_nll": 2.036898056666056,
      "candidate_id": "two_channel_operator",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.6633333365122477,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "squared_norm",
        "state": "paired_real",
        "training": "adamw",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.8770000139872233,
      "parameter_count": 19975.0,
      "parents": [
        "real_operator"
      ],
      "pareto": true,
      "shifted_ece": 0.17353104386064744,
      "shifted_nll": 1.805884904331631,
      "shifted_top1": 0.49877778026792735,
      "source_experiment": "QN-000014",
      "training_seconds": 3.1609636249995674
    },
    {
      "ambiguity_nll": 1.4280352592468262,
      "candidate_id": "real_accumulator",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 1,
      "genome": {
        "family": "accumulator",
        "measurement": "affine",
        "state": "real",
        "training": "adamw",
        "transition": "additive_commutative"
      },
      "in_domain_top1": 0.7366666793823242,
      "parameter_count": 19988.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.15008361637592316,
      "shifted_nll": 1.8522739542855158,
      "shifted_top1": 0.429222219520145,
      "source_experiment": "QN-000016",
      "training_seconds": 0.22708059700138014
    },
    {
      "ambiguity_nll": 2.4216166337331138,
      "candidate_id": "complex_accumulator",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 1,
      "genome": {
        "family": "accumulator",
        "measurement": "born",
        "state": "complex",
        "training": "adamw",
        "transition": "additive_commutative"
      },
      "in_domain_top1": 0.7036666671435038,
      "parameter_count": 19982.0,
      "parents": [
        "complex_operator"
      ],
      "pareto": false,
      "shifted_ece": 0.11020075115892623,
      "shifted_nll": 1.9706595871183605,
      "shifted_top1": 0.4151111112700568,
      "source_experiment": "QN-000016",
      "training_seconds": 0.5311076249975789
    },
    {
      "ambiguity_nll": 2.466770887374878,
      "candidate_id": "complex_magnitude_readout",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.825000007947286,
      "generation": 1,
      "genome": {
        "family": "operator",
        "measurement": "magnitude_only",
        "state": "complex",
        "training": "adamw",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.8610000014305115,
      "parameter_count": 20304.0,
      "parents": [
        "complex_operator"
      ],
      "pareto": false,
      "shifted_ece": 0.2657272285885281,
      "shifted_nll": 1.8382236825095284,
      "shifted_top1": 0.54288888308737,
      "source_experiment": "QN-000016",
      "training_seconds": 4.5210671246653265
    },
    {
      "ambiguity_nll": 3.2736082871754966,
      "candidate_id": "complex_no_negative",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.9716666539510092,
      "generation": 1,
      "genome": {
        "family": "operator",
        "measurement": "born_no_negative",
        "state": "complex",
        "training": "adamw",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.9153333306312561,
      "parameter_count": 20304.0,
      "parents": [
        "complex_operator"
      ],
      "pareto": true,
      "shifted_ece": 0.18953522708680895,
      "shifted_nll": 1.6729456583658855,
      "shifted_top1": 0.5748888850212097,
      "source_experiment": "QN-000016",
      "training_seconds": 4.215365555668541
    },
    {
      "ambiguity_nll": 1.435399015744527,
      "candidate_id": "density_rank1",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.675000011920929,
      "generation": 1,
      "genome": {
        "family": "density",
        "measurement": "diagonal",
        "state": "density_complex_rank1",
        "training": "adamw",
        "transition": "hamiltonian_dissipative"
      },
      "in_domain_top1": 0.8953333497047424,
      "parameter_count": 12920.0,
      "parents": [
        "density_rank2"
      ],
      "pareto": false,
      "shifted_ece": 0.2076826641956965,
      "shifted_nll": 2.0284685558742948,
      "shifted_top1": 0.4492222236262427,
      "source_experiment": "QN-000016",
      "training_seconds": 3.44627991666736
    },
    {
      "ambiguity_nll": 1.513328234354655,
      "candidate_id": "density_rank4",
      "compute_measure": "CPU training seconds",
      "context": "architecture",
      "counterfactual_pair_accuracy": 0.4283333321412404,
      "generation": 1,
      "genome": {
        "family": "density",
        "measurement": "diagonal",
        "state": "density_complex_rank4",
        "training": "adamw",
        "transition": "hamiltonian_dissipative"
      },
      "in_domain_top1": 0.8346666495005289,
      "parameter_count": 22880.0,
      "parents": [
        "density_rank2"
      ],
      "pareto": false,
      "shifted_ece": 0.22623832358254325,
      "shifted_nll": 2.1056300004323325,
      "shifted_top1": 0.4411111109786563,
      "source_experiment": "QN-000016",
      "training_seconds": 6.376674444332214
    },
    {
      "ambiguity_nll": 2.3678320248921714,
      "candidate_id": "complex_operator::adamw",
      "compute_measure": "CPU training seconds",
      "context": "training_law",
      "counterfactual_pair_accuracy": 1.0,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "backprop",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.9712499976158142,
      "parameter_count": 20304.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.2567332718107435,
      "shifted_nll": 1.6182807021670873,
      "shifted_top1": 0.6199999981456332,
      "source_experiment": "QN-000021",
      "training_seconds": 2.9575378889979524
    },
    {
      "ambiguity_nll": 2.3619795640309653,
      "candidate_id": "complex_operator::gradient_accumulation",
      "compute_measure": "CPU training seconds",
      "context": "training_law",
      "counterfactual_pair_accuracy": 1.0,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "backprop_accumulated",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.9712499976158142,
      "parameter_count": 20304.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.2585427910089493,
      "shifted_nll": 1.6134317186143663,
      "shifted_top1": 0.6223611103163825,
      "source_experiment": "QN-000021",
      "training_seconds": 3.909153527332819
    },
    {
      "ambiguity_nll": 1.317440390586853,
      "candidate_id": "complex_operator::hybrid_local_global",
      "compute_measure": "CPU training seconds",
      "context": "training_law",
      "counterfactual_pair_accuracy": 0.9911111195882162,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "local_then_backprop",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.9979166587193807,
      "parameter_count": 20304.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.17259996467166475,
      "shifted_nll": 2.3368953333960643,
      "shifted_top1": 0.41944444510671824,
      "source_experiment": "QN-000021",
      "training_seconds": 3.741594403004759
    },
    {
      "ambiguity_nll": 2.009526332219442,
      "candidate_id": "complex_operator::local_plasticity",
      "compute_measure": "CPU training seconds",
      "context": "training_law",
      "counterfactual_pair_accuracy": 0.21333333353201547,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "transition_local",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.6424999833106995,
      "parameter_count": 20304.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.023688283128043015,
      "shifted_nll": 2.9610698488023544,
      "shifted_top1": 0.13722222381167945,
      "source_experiment": "QN-000021",
      "training_seconds": 1.2473254443296657
    },
    {
      "ambiguity_nll": 2.363042672475179,
      "candidate_id": "complex_operator::multiobjective_adamw",
      "compute_measure": "CPU training seconds",
      "context": "training_law",
      "counterfactual_pair_accuracy": 1.0,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "backprop_auxiliary",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.9783333539962769,
      "parameter_count": 20304.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.2416675173574024,
      "shifted_nll": 1.5989308489693537,
      "shifted_top1": 0.6352777745988634,
      "source_experiment": "QN-000021",
      "training_seconds": 2.960437986332787
    },
    {
      "ambiguity_nll": 2.3891441027323403,
      "candidate_id": "complex_operator::pcgrad",
      "compute_measure": "CPU training seconds",
      "context": "training_law",
      "counterfactual_pair_accuracy": 1.0,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "projected_multitask_backprop",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.9783333341280619,
      "parameter_count": 20304.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.24166217115190294,
      "shifted_nll": 1.5963713857862685,
      "shifted_top1": 0.6349999904632568,
      "source_experiment": "QN-000021",
      "training_seconds": 5.477079083332986
    },
    {
      "ambiguity_nll": 2.340202252070109,
      "candidate_id": "complex_operator::phase_gradient",
      "compute_measure": "CPU training seconds",
      "context": "training_law",
      "counterfactual_pair_accuracy": 1.0,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "phase_rotated_multitask_backprop",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.9779166579246521,
      "parameter_count": 20304.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.24570544560750326,
      "shifted_nll": 1.5910786920123627,
      "shifted_top1": 0.6330555544959174,
      "source_experiment": "QN-000021",
      "training_seconds": 5.428970625333022
    },
    {
      "ambiguity_nll": 2.400062322616577,
      "candidate_id": "complex_operator::sgd",
      "compute_measure": "CPU training seconds",
      "context": "training_law",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "backprop",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.2383333295583725,
      "parameter_count": 20304.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.04921238455507491,
      "shifted_nll": 2.4317071437835693,
      "shifted_top1": 0.18777777751286825,
      "source_experiment": "QN-000021",
      "training_seconds": 2.905787152669897
    },
    {
      "ambiguity_nll": 2.9237112998962402,
      "candidate_id": "complex_operator::zerobackprop",
      "compute_measure": "CPU training seconds",
      "context": "training_law",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "operator",
        "measurement": "born",
        "state": "complex",
        "training": "frozen_state_centroid",
        "transition": "low_rank_noncommutative"
      },
      "in_domain_top1": 0.13291666905085245,
      "parameter_count": 20304.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.07865336785713832,
      "shifted_nll": 2.915938748253716,
      "shifted_top1": 0.13930555681387582,
      "source_experiment": "QN-000021",
      "training_seconds": 0.10512120833300287
    },
    {
      "ambiguity_nll": null,
      "candidate_id": "adaptive_attractor::fixed_final",
      "compute_measure": "CPU inference seconds per case",
      "context": "halting",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "attractor",
        "measurement": "distance_energy",
        "state": "real",
        "training": "frozen_checkpoint",
        "transition": "energy_descent_soft_act"
      },
      "in_domain_top1": 0.721666673819224,
      "parameter_count": 19801.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.3958573010232714,
      "shifted_nll": 2.964846054712931,
      "shifted_top1": 0.43133333656522965,
      "source_experiment": "QN-000023",
      "training_seconds": 1.3003087022257712e-05
    },
    {
      "ambiguity_nll": null,
      "candidate_id": "adaptive_attractor::hard",
      "compute_measure": "CPU inference seconds per case",
      "context": "halting",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 1,
      "genome": {
        "family": "attractor",
        "measurement": "distance_energy",
        "state": "real",
        "training": "frozen_checkpoint",
        "transition": "energy_descent_fixed2"
      },
      "in_domain_top1": 0.722000002861023,
      "parameter_count": 19801.0,
      "parents": [],
      "pareto": true,
      "shifted_ece": 0.1720769057671229,
      "shifted_nll": 2.127030107710097,
      "shifted_top1": 0.4324444499280718,
      "source_experiment": "QN-000023",
      "training_seconds": 3.4587370443558836e-06
    },
    {
      "ambiguity_nll": null,
      "candidate_id": "adaptive_attractor::soft",
      "compute_measure": "CPU inference seconds per case",
      "context": "halting",
      "counterfactual_pair_accuracy": 0.0,
      "generation": 0,
      "genome": {
        "family": "attractor",
        "measurement": "distance_energy",
        "state": "real",
        "training": "frozen_checkpoint",
        "transition": "energy_descent_soft_act"
      },
      "in_domain_top1": 0.7239999969800314,
      "parameter_count": 19801.0,
      "parents": [],
      "pareto": false,
      "shifted_ece": 0.32969709237416583,
      "shifted_nll": 2.5179518858591714,
      "shifted_top1": 0.4306666685475244,
      "source_experiment": "QN-000023",
      "training_seconds": 1.7121531466546005e-05
    }
  ],
  "claims": [
    {
      "claim": "The evaluated complex operator has an exact structured real-block implementation that reproduces its top-1 predictions at numerical precision.",
      "confidence": "high",
      "counterevidence": "This does not imply that every complex-valued architecture or optimization trajectory is equivalent to every unconstrained real network. Different parameterizations can impose different inductive biases even when the represented mapped function is the same. AMENDMENT-001: the two implementations do not merely represent the same function class; they share the same real parameter coordinates. ComplexOperatorState stores only real-valued nn.Parameters and constructs complex tensors inside the forward pass, so the parameter map between the two models is the identity and the observed agreement is a numerical-implementation result rather than a coordinate-transport result.",
      "evidence": "QN-000033 mapped mechanism runs differ by at most approximately 4.8e-7 across recorded metrics. QN-000042 complex and exact-real top-1 are identical in all 1,920 held-out nested cells; maximum NLL difference is 3.58e-7.",
      "status": "supported within implementation"
    },
    {
      "claim": "Within the reduced independent-task studies, the cellwise best-real envelope removes the observed complex robustness advantage.",
      "confidence": "high within tested scope",
      "counterevidence": "The independent discovery and confirmation profiles are outcome-ineligible and compute-reduced. The complete 14-model real envelope and full ShiftGauntlet outcome grid were not run. AMENDMENT-001: the 1,920 held-out cells are not 1,920 independent architecture comparisons. The cellwise best-real winner was exact_real_block_operator in 1,478 cells (77%) with mean effect exactly +0.00000 and 1,478 exact zeros, and real_polar_operator in 442 cells (23%) with mean -0.03978. The 1,478 zeros are equivalence-induced ties with an implementation that shares the complex model's parameter coordinates, not independent wins by a distinct real architecture. The entire -0.00916 mean is carried by the 442 real_polar_operator cells. AMENDMENT-001: QN-000040 is the healthier heterogeneous-control result. There state_space (637 cells, -0.09063), real_polar_operator (485, -0.04687), and gru (367, -0.07073) win 52% of discovery cells against exact_real_block_operator's 1,391.",
      "evidence": "QN-000040 has zero positive complex-minus-best-real effects among 2,880 nested discovery cells. QN-000042 has zero positive effects among 1,920 held-out nested cells and a family/world/seed hierarchical mean of -0.00916 with 95% interval [-0.01325, -0.00457].",
      "status": "falsified"
    },
    {
      "claim": "The frozen quadratic relationship selected on reduced discovery data fails quantitative prediction on held-out task families.",
      "confidence": "high",
      "counterevidence": "Effect-sign accuracy is 1.0 because all observed and predicted gaps are non-positive.",
      "evidence": "QN-LAW-001 discovery R2 is 0.949 with MAE 0.00260 across 12 aggregate cells. QN-000042 held-out R2 is -30.94 and MAE 0.0313; the frozen requirements were R2 at least 0.50 and MAE at most 0.015.",
      "status": "falsified"
    },
    {
      "claim": "The earlier positive NeuroWorld result is comparator-dependent rather than evidence of intrinsically complex arithmetic.",
      "confidence": "medium-high",
      "counterevidence": "The later studies use different generator families and reduced profiles, so the four point estimates are not one meta-analysis.",
      "evidence": "QN-000008 reports +0.0602 moderate-shift top-1 versus a two-channel real control. Later exact-real and best-real comparisons reverse the sign in QN-000031, QN-000040, and QN-000042.",
      "status": "bounded reinterpretation"
    },
    {
      "claim": "QN-GRAND-001 did not execute and provides no primary confirmatory effect estimate.",
      "confidence": "high",
      "counterevidence": "Eight readiness checks passed, including the shortcut gate, power plan, exact-real equivalence, and held-out family separation.",
      "evidence": "The registered preflight records six blocking failures and sealed_benchmark_opened=false.",
      "status": "not executed"
    },
    {
      "claim": "The repository contains no evidence of clinical validity, patient benefit, safety, or a quantum mechanism of cognition.",
      "confidence": "high",
      "counterevidence": "",
      "evidence": "All next-phase tasks are synthetic and nonclinical; no patient dataset was selected or evaluated.",
      "status": "unsupported"
    }
  ],
  "experiments": [
    {
      "artifact_count": 3,
      "id": "QN-000001",
      "status": "complete"
    },
    {
      "artifact_count": 4,
      "id": "QN-000002",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000003",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000004",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000005",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000006",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000007",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000008",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000009",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000010",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000011",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000012",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000013",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000014",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000015",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000016",
      "status": "complete"
    },
    {
      "artifact_count": 4,
      "id": "QN-000017",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000018",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000019",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000020",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000021",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000022",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000023",
      "status": "complete"
    },
    {
      "artifact_count": 4,
      "id": "QN-000025",
      "status": "complete"
    },
    {
      "artifact_count": 7,
      "id": "QN-000026",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000027",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000028",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000029",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000030",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000031",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000032",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000033",
      "status": "complete"
    },
    {
      "artifact_count": 4,
      "id": "QN-000034",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000035",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000036",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000037",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000039",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000040",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000041",
      "status": "complete"
    },
    {
      "artifact_count": 3,
      "id": "QN-000042",
      "status": "complete"
    },
    {
      "artifact_count": 4,
      "id": "QN-GRAND-001",
      "status": "blocked_before_execution"
    }
  ],
  "failures": [
    "Metadata and unintended order-token shortcuts crossed frozen thresholds.",
    "Smoke profile produced too few law cells for the frozen candidate fitter.",
    "QN-LAW-001 failed R2 and MAE thresholds on untouched task families.",
    "Six mandatory readiness gates failed before sealed benchmark access.",
    "The best-real envelope eliminates the complex advantage, and exact-real computation matches complex predictions."
  ],
  "falsification": [
    {
      "comparator": "two-channel real",
      "label": "initial NeuroWorld",
      "value": 0.06016665697097778
    },
    {
      "comparator": "best evaluated real",
      "label": "power pilot",
      "value": -0.05519245126334431
    },
    {
      "comparator": "cellwise best real",
      "label": "reduced discovery",
      "value": -0.03695312475028913
    },
    {
      "comparator": "cellwise best real",
      "label": "held-out families",
      "value": -0.009157985999869804
    }
  ],
  "frontiers": {
    "architecture": [
      "adaptive_attractor",
      "complex_no_negative",
      "complex_operator",
      "logistic",
      "mlp",
      "real_accumulator",
      "real_operator",
      "state_space",
      "transformer",
      "two_channel_operator"
    ],
    "halting": [
      "adaptive_attractor::hard"
    ],
    "training_law": [
      "complex_operator::adamw",
      "complex_operator::local_plasticity",
      "complex_operator::multiobjective_adamw",
      "complex_operator::pcgrad",
      "complex_operator::phase_gradient",
      "complex_operator::sgd",
      "complex_operator::zerobackprop"
    ]
  },
  "generated_from": [
    "QN-000008",
    "QN-000014",
    "QN-000016",
    "QN-000021",
    "QN-000023",
    "QN-000025",
    "QN-000026",
    "QN-000031",
    "QN-000033",
    "QN-000040",
    "QN-000042",
    "QN-GRAND-001"
  ],
  "grand_preflight": {
    "all_required_gates_pass": false,
    "blocking_failures": [
      "complete_preregistered_real_envelope",
      "equal_or_real_favoring_search_budgets",
      "compute_matching_records",
      "full_shiftgauntlet_outcome_grid",
      "full_discovery_protocol",
      "raw_predictions_preserved"
    ],
    "checks": [
      {
        "blocking": true,
        "check_id": "exact_real_equivalence",
        "evidence": "QN-000042 max top1 difference=0; max NLL difference=3.58e-07",
        "passed": true
      },
      {
        "blocking": true,
        "check_id": "evaluated_parameter_matching",
        "evidence": "Evaluated model parameter counts: {'complex_operator': {'name': 'complex_operator', 'parameter_budget': 20304, 'parameter_count': 20304, 'rank': 2, 'state_real_dof': 48, 'trainable_parameter_count': 20304, 'width': 24}, 'exact_real_block_operator': {'name': 'exact_real_block_operator', 'parameter_budget': 20304, 'parameter_count': 20304, 'rank': 2, 'state_real_dof': 48, 'trainable_parameter_count': 20304, 'width': 24}, 'state_space': {'name': 'state_space', 'parameter_budget': 20304, 'parameter_count': 20285, 'rank': 0, 'state_real_dof': 193, 'trainable_parameter_count': 20285, 'width': 193}, 'gru': {'name': 'gru', 'parameter_budget': 20304, 'parameter_count': 20390, 'rank': 0, 'state_real_dof': 60, 'trainable_parameter_count': 20390, 'width': 60}, 'real_polar_operator': {'name': 'real_polar_operator', 'parameter_budget': 20304, 'parameter_count': 20304, 'rank': 2, 'state_real_dof': 48, 'trainable_parameter_count': 20304, 'width': 24}}",
        "passed": true
      },
      {
        "blocking": true,
        "check_id": "simulator_red_team",
        "evidence": "QN-000028 frozen shortcut gate",
        "passed": true
      },
      {
        "blocking": true,
        "check_id": "shift_gauntlet_structure",
        "evidence": "QN-000029 structural ShiftGauntlet audit",
        "passed": true
      },
      {
        "blocking": true,
        "check_id": "power_target",
        "evidence": "QN-000031 selected 32 worlds at estimated power 1.0",
        "passed": true
      },
      {
        "blocking": true,
        "check_id": "law_frozen_before_heldout_confirmation",
        "evidence": "QN-LAW-001 source QN-000040; held-out evaluation QN-000042",
        "passed": true
      },
      {
        "blocking": true,
        "check_id": "untouched_nonmedical_family",
        "evidence": "Held-out families: ['bayesian_urn', 'hidden_rule_relational', 'machine_fault_diagnosis', 'network_intrusion_reasoning']",
        "passed": true
      },
      {
        "blocking": true,
        "check_id": "clean_confirmatory_worktree",
        "evidence": "QN-000042 environment.json",
        "passed": true
      },
      {
        "blocking": true,
        "check_id": "complete_preregistered_real_envelope",
        "evidence": "Evaluated 4 of 14 required real controls; missing=['causal_transformer', 'dense_real_recurrence', 'lstm', 'orthogonal_real_recurrence', 'real_operator', 'real_rotation_block_operator', 'residual_gated_recurrence', 'transformer_encoder', 'unrestricted_paired_real_operator', 'vanilla_rnn']",
        "passed": false
      },
      {
        "blocking": true,
        "check_id": "equal_or_real_favoring_search_budgets",
        "evidence": "Observed maximum tuning trials per evaluated model: {'complex_operator': 1, 'exact_real_block_operator': 1, 'state_space': 1, 'gru': 1, 'real_polar_operator': 1}",
        "passed": false
      },
      {
        "blocking": true,
        "check_id": "compute_matching_records",
        "evidence": "Per-trial FLOPs and optimizer-step records required by QNF-PREREG-002 are absent.",
        "passed": false
      },
      {
        "blocking": true,
        "check_id": "full_shiftgauntlet_outcome_grid",
        "evidence": "QN-000029 is structural and QN-000031 is a reduced pilot; no full confirmatory grid.",
        "passed": false
      },
      {
        "blocking": true,
        "check_id": "full_discovery_protocol",
        "evidence": "QN-000040 deviations=['Training sizes 5000 and 25000 are deferred; only 250 and 1000 are evaluated.', 'The full real envelope and 8/10-trial hyperparameter policy are not executed.', 'Results are provisional discovery evidence and cannot support Outcome E or QN-GRAND-001.']; frozen scope forbids grand use.",
        "passed": false
      },
      {
        "blocking": true,
        "check_id": "raw_predictions_preserved",
        "evidence": "Prediction artifacts found: []",
        "passed": false
      }
    ],
    "evidence_sha256": {
      "experiments/results/QN-000028/metrics.json": "4486a29fbea3708361b941b3d89f442e7730ddb2b6fa92de59c5a1de767cb3f0",
      "experiments/results/QN-000029/metrics.json": "b050b669607ae2705aacc78d109bc5e119610cb656c5301b694593e7b5545eb9",
      "experiments/results/QN-000031/metrics.json": "f5d4c3e1816ab2e6fb15ee1263d8adb39754091033d30dd231d8bb04043b85b4",
      "experiments/results/QN-000033/metrics.json": "a0acf17b778c7cb213cff0d20dd278b39e5d7db5fa6bfc94da35312211aa3a58",
      "experiments/results/QN-000040/metrics.json": "d06945ea7257b12c8b515c3f361e9e31925554ea5ee8a27e36ba65e3ddc1428b",
      "experiments/results/QN-000042/metrics.json": "3106fdbf324e80903eb1f6eec19a0db20f6c3eb45abdd49c58d58e26ab792d52"
    },
    "experiment_id": "QN-GRAND-001",
    "sealed_benchmark_opened": false,
    "status": "blocked"
  },
  "historical_claims": [
    {
      "claim": "The implemented complex operator has an exact structured-real equivalent",
      "confidence": "High",
      "counterevidence": "Does not equate every possible complex and real network or optimizer",
      "evidence": "QN-000033 and QN-000042; held-out top-1 difference is zero in 1,920 nested cells",
      "status": "Supported within implementation"
    },
    {
      "claim": "Complex retains a best-real robustness advantage on independent tasks",
      "confidence": "High within reduced scope",
      "counterevidence": "QN-000040 and QN-000042 contain zero positive nested effects; held-out hierarchical mean \u22120.00916",
      "evidence": "None",
      "status": "Falsified"
    },
    {
      "claim": "QN-LAW-001 generalizes quantitatively",
      "confidence": "High",
      "counterevidence": "Held-out R2 \u221230.94 and MAE 0.0313 fail frozen thresholds",
      "evidence": "Sign transfers",
      "status": "Falsified"
    },
    {
      "claim": "QN-GRAND-001 provides confirmatory evidence",
      "confidence": "High",
      "counterevidence": "Six mandatory preflight gates failed; sealed benchmark unopened",
      "evidence": "None",
      "status": "Not executed"
    },
    {
      "claim": "Q-Neuro has clinical validity or reveals quantum cognition",
      "confidence": "High",
      "counterevidence": "Synthetic and nonclinical evidence only",
      "evidence": "None",
      "status": "Unsupported"
    },
    {
      "claim": "Ordered computation is required for the chronology-twin task",
      "confidence": "High",
      "counterevidence": "This is true by task construction and does not establish a broad medical phenomenon",
      "evidence": "QN-000003: ordered models 1.0 pair accuracy; MLP 0.0; twin vectors are identical by construction",
      "status": "Replicated in simulator"
    },
    {
      "claim": "Low-rank operator states are more sample-efficient than the tested tiny Transformer",
      "confidence": "Medium for the narrow Transformer comparison",
      "counterevidence": "QN-000006 tuned GRU strongly exceeds operators at 250 cases",
      "evidence": "QN-000004 and QN-000006: real/complex exceed Transformer through 1,000 cases",
      "status": "Preliminary"
    },
    {
      "claim": "Operators are the most sample-efficient in-domain mechanism",
      "confidence": "High",
      "counterevidence": "QN-000006 tuned GRU reaches 0.920 at 250 cases versus 0.774 real and 0.699 complex",
      "evidence": "QN-000004 initially suggested this",
      "status": "Refuted under tested setup"
    },
    {
      "claim": "The complex model uses relative phase",
      "confidence": "High for dependence, low for benefit",
      "counterevidence": "Post-training ablation disrupts the representation; a trained phase-free equivalent is stronger evidence",
      "evidence": "QN-000003: zero/random phase top-1 near 0.21 versus 1.0 learned",
      "status": "Preliminary"
    },
    {
      "claim": "Complex operators improve top-1 after 500 cases",
      "confidence": "Low\u2013medium",
      "counterevidence": "Real wins at 250; complex NLL/ECE/runtime are worse; task saturates; no two-channel real control",
      "evidence": "QN-000004: complex exceeds real from 500\u20135,000 cases across three seeds",
      "status": "Preliminary"
    },
    {
      "claim": "Complex operators are more robust to declared NeuroWorld shifts than the tested controls",
      "confidence": "Medium",
      "counterevidence": "Worlds share one simulator family; shifts were project-designed; no external data",
      "evidence": "QN-000008: complex beats every control across five unseen worlds and four severities; world-level paired CIs exclude zero",
      "status": "Replicated in simulator"
    },
    {
      "claim": "Complex structure adds robustness beyond the tested two-channel real control",
      "confidence": "Medium-low",
      "counterevidence": "Control is not algebraically exhaustive; complex calibration is not better",
      "evidence": "QN-000008 complex-minus-two-channel top-1 is +0.054 to +0.063 across severities with five-world CIs above zero",
      "status": "Replicated in simulator"
    },
    {
      "claim": "In-domain temperature calibration transfers under shift",
      "confidence": "High under tested setup",
      "counterevidence": "QN-000008: validation-fitted scaling worsens shifted ECE for every model and can severely worsen NLL",
      "evidence": "None",
      "status": "Refuted"
    },
    {
      "claim": "Complex dynamics improve held-out evidence composition",
      "confidence": "Low",
      "counterevidence": "Two-channel also reaches 1.000; all operator/GRU models are at 0.995 or higher; task saturates",
      "evidence": "QN-000010: complex reaches 1.000 top-1",
      "status": "Not supported"
    },
    {
      "claim": "Complex hypothesis states represent irreducible ambiguity better",
      "confidence": "High for this task",
      "counterevidence": "QN-000010: complex pair NLL 2.581 versus 1.148 real and 1.453 two-channel; valid-twin mass only 0.212",
      "evidence": "None",
      "status": "Refuted under tested setup"
    },
    {
      "claim": "The complex output score uniquely enables unknown-disease rejection",
      "confidence": "Low",
      "counterevidence": "Two-channel is 0.9974 with paired CI for the difference crossing zero; GRU is 0.9847",
      "evidence": "QN-000010: complex MSP AUROC 0.9988",
      "status": "Not supported"
    },
    {
      "claim": "Complex representation geometry robustly separates the hidden syndrome",
      "confidence": "Medium-low",
      "counterevidence": "Only one hand-designed syndrome; real operator also reaches 0.9918; three seeds; separation is not attractor discovery",
      "evidence": "QN-000010: centroid-distance AUROC 0.9990 \u00b1 0.0006; higher and far more stable than two-channel",
      "status": "Preliminary"
    },
    {
      "claim": "Q-Neuro discovers an unseen disease attractor",
      "confidence": "Very low",
      "counterevidence": "No unsupervised cluster-number recovery, attractor dynamics, or prospective assignment experiment",
      "evidence": "Hidden syndrome is strongly separable in QN-000010",
      "status": "Unsupported"
    },
    {
      "claim": "Complex expected-information acquisition is more evidence-efficient than random/fixed querying",
      "confidence": "Medium for within-model policy effect",
      "counterevidence": "MLP EIG is 0.585 and two-channel is 0.568; complex differences against both have intervals crossing zero; EIG costs more runtime",
      "evidence": "QN-000012: complex accuracy AUC 0.590 vs 0.517 fixed and 0.463 random; paired EIG-minus-fixed interval is positive",
      "status": "Preliminary"
    },
    {
      "claim": "Hypothesis-state architectures are uniquely suited to active diagnosis",
      "confidence": "Low",
      "counterevidence": "MLP is statistically indistinguishable; real EIG is worse than its fixed order; task is synthetic and binary",
      "evidence": "Complex is strongest by mean AUC in QN-000012",
      "status": "Not supported"
    },
    {
      "claim": "Expected information gain always improves evidence acquisition",
      "confidence": "High",
      "counterevidence": "Harms Transformer by \u22120.169 AUC vs fixed; negligible for real; seed-variable for GRU/two-channel",
      "evidence": "Helps complex and MLP in QN-000012",
      "status": "Refuted"
    },
    {
      "claim": "Full-information accuracy predicts evidence efficiency",
      "confidence": "High",
      "counterevidence": "QN-000012 GRU/Transformer retain 0.950/0.982 full accuracy but have 0.282/0.359 EIG AUC",
      "evidence": "Full top-1 is high for every model",
      "status": "Refuted"
    },
    {
      "claim": "Coherent Hamiltonian-style evolution is more robust than dissipative-only evolution",
      "confidence": "Medium-low",
      "counterevidence": "Three worlds in one simulator family; different learned widths despite parameter matching; no external task",
      "evidence": "QN-000014: 0.556 vs 0.438 moderate-shift top-1; hybrid-minus-dissipative world effect +0.112 with interval above zero",
      "status": "Preliminary"
    },
    {
      "claim": "Adding dissipation improves Hamiltonian dynamics",
      "confidence": "Medium under tested setup",
      "counterevidence": "QN-000014 hybrid 0.550 vs Hamiltonian 0.556; paired interval includes zero",
      "evidence": "None",
      "status": "Not supported"
    },
    {
      "claim": "Adaptive diagnostic time reduces compute without performance loss",
      "confidence": "Medium for realized truncation, none for case adaptivity",
      "counterevidence": "Every case stops at step two, so a fixed shallow model is equivalent; timing is CPU-specific; model remains weak",
      "evidence": "QN-000023 hard exit uses 2/8 states, 20.2% of soft-path CPU latency, and unchanged top-1",
      "status": "Partially supported"
    },
    {
      "claim": "D3 off-diagonal state improves later diagnostic resolution",
      "confidence": "Very low",
      "counterevidence": "Later-resolution prediction was not tested; D3 trails real on shift and does not improve ambiguity NLL",
      "evidence": "D3 has valid density state and mean off-diagonal coherence 0.620",
      "status": "Unresolved"
    },
    {
      "claim": "The factor-graph prior improves diagnosis",
      "confidence": "High",
      "counterevidence": "QN-000014 GNN top-1 0.319 ID and 0.184 shifted, substantially below simple controls",
      "evidence": "Declared NeuroWorld causal groups define edges",
      "status": "Refuted for current graph model"
    },
    {
      "claim": "Ordered state-conditioned complex composition contributes to robustness",
      "confidence": "Medium-low",
      "counterevidence": "Accumulator lacks both non-commutativity and multiplicative state conditioning, so the precise cause is not isolated",
      "evidence": "QN-000016 full complex exceeds commutative complex accumulator by +0.232 shifted top-1 and +1.000 pair accuracy",
      "status": "Preliminary"
    },
    {
      "claim": "Phase-sensitive readout interference contributes to robustness",
      "confidence": "Medium-low",
      "counterevidence": "Evolution remains complex in both; readout changes optimization geometry; synthetic only",
      "evidence": "QN-000016 full complex exceeds magnitude-only constructive readout by +0.104 across worlds",
      "status": "Preliminary"
    },
    {
      "claim": "Explicit negative evidence contributes to robustness",
      "confidence": "Medium",
      "counterevidence": "Ablation drops information rather than replacing it with a calibrated contradiction law",
      "evidence": "QN-000016 removing negative tokens costs 0.072 shifted top-1 and worsens ambiguity/pairs",
      "status": "Preliminary"
    },
    {
      "claim": "Higher density rank improves D3",
      "confidence": "Medium",
      "counterevidence": "QN-000016 ranks 1/2/4 obtain 0.449/0.453/0.441 shift top-1, with worsening pair accuracy as rank increases",
      "evidence": "None",
      "status": "Refuted under tested setup"
    },
    {
      "claim": "Diagnosis-trained complex states expose simulator hierarchy",
      "confidence": "Medium for frozen-state accessibility",
      "counterevidence": "Labels are synthetic; extractability does not establish use, disentanglement, or causal semantics",
      "evidence": "QN-000019 linear probes recover mechanism/localization/temporality/context at 0.932/0.933/0.907/0.918",
      "status": "Preliminary"
    },
    {
      "claim": "Complex states uniquely encode hierarchical factors",
      "confidence": "High under tested probe suite",
      "counterevidence": "GRU and state-space probes are generally stronger; complex-minus-GRU is negative on every factor",
      "evidence": "Complex probes are accurate in QN-000019",
      "status": "Refuted"
    },
    {
      "claim": "Hermitian observables improve interpretable complex-state measurement",
      "confidence": "Low",
      "counterevidence": "Most NLL effects worsen; only three seeds; a learned quadratic classifier is not inherently interpretable",
      "evidence": "QN-000019 quadratic probes improve complex accuracy, especially temporality (+0.039) and context (+0.023)",
      "status": "Preliminary"
    },
    {
      "claim": "Hierarchical probe accuracy predicts shift robustness",
      "confidence": "Low",
      "counterevidence": "Descriptive correlation over 18 non-independent architectures; GRU is the strongest probe state but among the weakest under shift",
      "evidence": "Cross-model Pearson `r=+0.45` in QN-000019",
      "status": "Not supported"
    },
    {
      "claim": "Phase Gradient Optimization improves multi-objective training",
      "confidence": "High for tested setup",
      "counterevidence": "Multi-objective AdamW is 0.002 higher in about half the time; gradients are weakly aligned; PGO receives extra factor labels",
      "evidence": "QN-000021 PGO is +0.013 shifted top-1 over diagnosis-only AdamW",
      "status": "Not supported"
    },
    {
      "claim": "Auxiliary mechanism/localization supervision improves complex-operator robustness",
      "confidence": "Medium-low",
      "counterevidence": "Small effect, only three worlds, synthetic factor labels, same simulator",
      "evidence": "QN-000021 multi-objective AdamW gains +0.015 shifted top-1 over AdamW across worlds",
      "status": "Preliminary"
    },
    {
      "claim": "Q-Neuro can learn useful diagnosis without global backpropagation",
      "confidence": "Low",
      "counterevidence": "Shift top-1 is 0.137, chronology pairs 0.213, and scaling from 250 to 1,000 does not help transfer",
      "evidence": "Local plasticity reaches 0.642 in-domain top-1 with zero backward calls",
      "status": "Preliminary source-only competence"
    },
    {
      "claim": "ZeroBackprop is competitive with end-to-end learning",
      "confidence": "High",
      "counterevidence": "QN-000021 top-1 is 0.133 in-domain and 0.139 shifted; chronology pair accuracy is zero",
      "evidence": "It fits a readout in 0.11 CPU seconds with zero gradients",
      "status": "Refuted for centroid prototype"
    },
    {
      "claim": "Local pretraining improves global generalization",
      "confidence": "High under tested setup",
      "counterevidence": "Shift top-1 is 0.419, 0.201 below AdamW across worlds; source specialization repeats at both train sizes",
      "evidence": "Hybrid reaches 0.998 in-domain and ambiguity NLL 1.317",
      "status": "Refuted"
    },
    {
      "claim": "Learned case-adaptive halting is useful",
      "confidence": "High under tested setup",
      "counterevidence": "QN-000023 sends 100% of cases to the same two-state boundary; a fixed-depth truncation is simpler",
      "evidence": "QN-000014 learns a soft expected depth and QN-000023 validates a velocity threshold",
      "status": "Not supported"
    },
    {
      "claim": "Later attractor iterations improve diagnostic refinement",
      "confidence": "Medium",
      "counterevidence": "QN-000023 eight-state final readout has the same top-1 but shifted NLL 2.965 vs 2.127 at two states",
      "evidence": "None",
      "status": "Refuted under tested setup"
    },
    {
      "claim": "Complex state trajectories visibly preserve evidence order",
      "confidence": "High for measured complex state",
      "counterevidence": "Synthetic construction deliberately makes order causal; conventional ordered models can also solve it",
      "evidence": "QN-000025 chronology pairs have 1.000 pair accuracy and normalized final-state distance 0.841 despite identical aggregate evidence",
      "status": "Replicated in simulator"
    },
    {
      "claim": "Complex dynamics support contradiction and revival",
      "confidence": "Low",
      "counterevidence": "Observed-negative tokens are not semantic contradictions; recovery is probability-based; only synthetic cases",
      "evidence": "QN-000025 finds >0.05 negative-token drops in 5.0% of cases and later recovery in 75.9% of those",
      "status": "Preliminary operational behavior"
    },
    {
      "claim": "Complex final states are stable attractors",
      "confidence": "Medium",
      "counterevidence": "QN-000025 mean final-token velocity is 0.175; trajectory endpoint depends on observation length",
      "evidence": "None",
      "status": "Not supported"
    },
    {
      "claim": "State trajectories are intrinsically interpretable",
      "confidence": "Low",
      "counterevidence": "No semantic axis validation or human study; visibility is not explanation or causal attribution",
      "evidence": "Probabilities, amplitudes, entropy, velocity, and paths are directly observable",
      "status": "Unsupported"
    },
    {
      "claim": "The discovery engine identifies a universally superior candidate",
      "confidence": "High",
      "counterevidence": "Fronts remain broad in every context; objectives and proposals are human-declared; outputs are decision support",
      "evidence": "QN-000026 computes explicit multi-objective Pareto fronts",
      "status": "Refuted by tradeoffs"
    },
    {
      "claim": "The QN-000026 proposals are scientific discoveries",
      "confidence": "None",
      "counterevidence": "No proposal has been tested; automated ranking cannot promote a hypothesis into evidence",
      "evidence": "They are reproducible transformations of registered measurements",
      "status": "Unsupported"
    },
    {
      "claim": "Complex arithmetic is superior overall",
      "confidence": "Low",
      "counterevidence": "GRU wins low-data in-domain; real is better calibrated; complex is slower; evidence is synthetic",
      "evidence": "Shift robustness supports one dimension",
      "status": "Not supported"
    },
    {
      "claim": "Q-Neuro is novel",
      "confidence": "Very low",
      "counterevidence": "Every major mechanism has close prior work; the review is not systematic and combination novelty is unproved",
      "evidence": "A primary-source mechanism review is documented in `docs/PRIOR_ART.md`",
      "status": "Not established"
    },
    {
      "claim": "Q-Neuro has clinical diagnostic value",
      "confidence": "None",
      "counterevidence": "No real data, external validation, prospective study, or medical-device evaluation",
      "evidence": "None; only synthetic archetypes tested",
      "status": "Unsupported"
    }
  ],
  "historical_failures": [
    "Full-data Experiment Zero as a discriminator",
    "Complex phase as an automatic advantage",
    "Asymmetric-input comparison (QN-000002)",
    "Operator states as the strongest low-data baseline",
    "In-domain temperature scaling as a shift-calibration fix",
    "Held-out composition at 3,000 cases as an architectural discriminator",
    "Complex states as automatic protection against premature collapse",
    "Expected information gain as a universally superior query rule",
    "Dissipation as an automatic diagnostic-elimination advantage",
    "Soft adaptive depth as compute savings",
    "Fixed NeuroWorld factor graph",
    "Density rank as useful relational capacity",
    "Complex state as a uniquely hierarchical representation",
    "Phase-coded gradients as a robustness optimizer",
    "Local pretraining as a benign initialization",
    "Centroid ZeroBackprop as an end-to-end alternative",
    "Final complex state as a converged disease attractor",
    "Automated Pareto ranking as a universal winner selector"
  ],
  "next_steps": [
    {
      "falsifier": "Must pass prospectively before any future sealed grand benchmark.",
      "id": "complete_preregistered_real_envelope",
      "mechanism": "Evaluated 4 of 14 required real controls; missing=['causal_transformer', 'dense_real_recurrence', 'lstm', 'orthogonal_real_recurrence', 'real_operator', 'real_rotation_block_operator', 'residual_gated_recurrence', 'transformer_encoder', 'unrestricted_paired_real_operator', 'vanilla_rnn']",
      "priority": "blocking"
    },
    {
      "falsifier": "Must pass prospectively before any future sealed grand benchmark.",
      "id": "equal_or_real_favoring_search_budgets",
      "mechanism": "Observed maximum tuning trials per evaluated model: {'complex_operator': 1, 'exact_real_block_operator': 1, 'state_space': 1, 'gru': 1, 'real_polar_operator': 1}",
      "priority": "blocking"
    },
    {
      "falsifier": "Must pass prospectively before any future sealed grand benchmark.",
      "id": "compute_matching_records",
      "mechanism": "Per-trial FLOPs and optimizer-step records required by QNF-PREREG-002 are absent.",
      "priority": "blocking"
    },
    {
      "falsifier": "Must pass prospectively before any future sealed grand benchmark.",
      "id": "full_shiftgauntlet_outcome_grid",
      "mechanism": "QN-000029 is structural and QN-000031 is a reduced pilot; no full confirmatory grid.",
      "priority": "blocking"
    },
    {
      "falsifier": "Must pass prospectively before any future sealed grand benchmark.",
      "id": "full_discovery_protocol",
      "mechanism": "QN-000040 deviations=['Training sizes 5000 and 25000 are deferred; only 250 and 1000 are evaluated.', 'The full real envelope and 8/10-trial hyperparameter policy are not executed.', 'Results are provisional discovery evidence and cannot support Outcome E or QN-GRAND-001.']; frozen scope forbids grand use.",
      "priority": "blocking"
    },
    {
      "falsifier": "Must pass prospectively before any future sealed grand benchmark.",
      "id": "raw_predictions_preserved",
      "mechanism": "Prediction artifacts found: []",
      "priority": "blocking"
    }
  ],
  "proposals": [
    {
      "evidence": "QN-000023 executes two states for every case with unchanged accuracy and better calibration.",
      "falsifier": "A retrained two-state model loses more than 0.01 source top-1 or 0.02 shifted top-1.",
      "id": "fixed_two_state_attractor",
      "mechanism": "Replace soft ACT and eight-state dynamics with two fixed energy steps.",
      "parent": "adaptive_attractor",
      "priority": "high"
    },
    {
      "evidence": "Complex is robust but has ambiguity NLL above 2.3 in multiple suites.",
      "falsifier": "Ambiguity NLL does not improve without more than 0.02 shifted top-1 loss.",
      "id": "ambiguity_aware_complex_measurement",
      "mechanism": "Add set-valued twin mass and calibration loss only on constructed ambiguous cases.",
      "parent": "complex_operator",
      "priority": "high"
    },
    {
      "evidence": "Local and hybrid learning lock into the source world in QN-000021.",
      "falsifier": "Shift top-1 remains below 0.40 at 1,000 cases.",
      "id": "multiworld_local_plasticity",
      "mechanism": "Drive local prototype updates with counterfactual and multi-world batches.",
      "parent": "local_plasticity",
      "priority": "medium"
    },
    {
      "evidence": "Hermitian probes improve accuracy but usually worsen NLL in QN-000019.",
      "falsifier": "Hermitian accuracy gain disappears or NLL remains worse than linear probes.",
      "id": "calibrated_hermitian_observables",
      "mechanism": "Fit temperature-regularized Hermitian probes with explicit validation calibration.",
      "parent": "complex_operator",
      "priority": "medium"
    },
    {
      "evidence": "PGO adds cost on weakly aligned tasks and does not beat multi-objective AdamW.",
      "falsifier": "On a genuinely conflicting task, it fails to beat the same-objective AdamW control.",
      "id": "conflict_conditioned_pgo",
      "mechanism": "Activate phase rotation only after measuring negative task-gradient cosine.",
      "parent": "phase_gradient",
      "priority": "low"
    }
  ],
  "robustness": [
    {
      "in_domain": 0.995667,
      "mild": 0.805967,
      "model": "complex_operator",
      "moderate": 0.645,
      "nuisance": 0.909167,
      "severe": 0.468167
    },
    {
      "in_domain": 0.982167,
      "mild": 0.336633,
      "model": "gru",
      "moderate": 0.2562,
      "nuisance": 0.372567,
      "severe": 0.180733
    },
    {
      "in_domain": 0.722333,
      "mild": 0.509767,
      "model": "mlp",
      "moderate": 0.3989,
      "nuisance": 0.584067,
      "severe": 0.288467
    },
    {
      "in_domain": 0.984333,
      "mild": 0.6723,
      "model": "real_operator",
      "moderate": 0.5225,
      "nuisance": 0.7883,
      "severe": 0.3769
    },
    {
      "in_domain": 0.9095,
      "mild": 0.631833,
      "model": "transformer",
      "moderate": 0.491233,
      "nuisance": 0.747467,
      "severe": 0.358567
    },
    {
      "in_domain": 0.986,
      "mild": 0.7451,
      "model": "two_channel_operator",
      "moderate": 0.584833,
      "nuisance": 0.846467,
      "severe": 0.414333
    }
  ],
  "surprises": [
    {
      "candidate_id": "complex_no_negative",
      "message": "High source accuracy coexists with poor irreducible-ambiguity NLL.",
      "severity": 3.2736082871754966,
      "type": "accuracy_ambiguity_tension"
    },
    {
      "candidate_id": "complex_operator::pcgrad",
      "message": "High source accuracy coexists with poor irreducible-ambiguity NLL.",
      "severity": 2.3891441027323403,
      "type": "accuracy_ambiguity_tension"
    },
    {
      "candidate_id": "complex_operator::adamw",
      "message": "High source accuracy coexists with poor irreducible-ambiguity NLL.",
      "severity": 2.3678320248921714,
      "type": "accuracy_ambiguity_tension"
    },
    {
      "candidate_id": "complex_operator::multiobjective_adamw",
      "message": "High source accuracy coexists with poor irreducible-ambiguity NLL.",
      "severity": 2.363042672475179,
      "type": "accuracy_ambiguity_tension"
    },
    {
      "candidate_id": "complex_operator::gradient_accumulation",
      "message": "High source accuracy coexists with poor irreducible-ambiguity NLL.",
      "severity": 2.3619795640309653,
      "type": "accuracy_ambiguity_tension"
    },
    {
      "candidate_id": "complex_operator",
      "message": "High source accuracy coexists with poor irreducible-ambiguity NLL.",
      "severity": 2.3517801761627197,
      "type": "accuracy_ambiguity_tension"
    },
    {
      "candidate_id": "complex_operator::phase_gradient",
      "message": "High source accuracy coexists with poor irreducible-ambiguity NLL.",
      "severity": 2.340202252070109,
      "type": "accuracy_ambiguity_tension"
    },
    {
      "candidate_id": "gru",
      "message": "High source accuracy coexists with poor irreducible-ambiguity NLL.",
      "severity": 2.295586188634237,
      "type": "accuracy_ambiguity_tension"
    },
    {
      "candidate_id": "gru",
      "message": "High source accuracy coexists with a large unseen-world collapse.",
      "severity": 0.7398888965447743,
      "type": "generalization_reversal"
    },
    {
      "candidate_id": "real_accumulator",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7366666793823242,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "mlp",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7263333201408386,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "logistic",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7249999841054281,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "adaptive_attractor",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7239999969800314,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "adaptive_attractor::soft",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7239999969800314,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "dissipative",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7226666808128357,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "adaptive_attractor::hard",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.722000002861023,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "adaptive_attractor::fixed_final",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.721666673819224,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "coupled_tensor",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7193333307902018,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "energy_attractor",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7169999877611796,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "complex_mlp",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7076666553815206,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "complex_accumulator",
      "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
      "severity": 0.7036666671435038,
      "type": "order_blind_success"
    },
    {
      "candidate_id": "state_space",
      "message": "High source accuracy coexists with a large unseen-world collapse.",
      "severity": 0.6402222149901919,
      "type": "generalization_reversal"
    },
    {
      "candidate_id": "complex_operator::hybrid_local_global",
      "message": "High source accuracy coexists with a large unseen-world collapse.",
      "severity": 0.5784722136126625,
      "type": "generalization_reversal"
    },
    {
      "candidate_id": "real_operator",
      "message": "High source accuracy coexists with a large unseen-world collapse.",
      "severity": 0.4815555645359887,
      "type": "generalization_reversal"
    },
    {
      "candidate_id": "adaptive_attractor::hard",
      "message": "All cases stop at two states; realized savings are not adaptive.",
      "severity": 0.75,
      "type": "hard_halting_degeneracy"
    },
    {
      "candidate_id": "complex_operator::phase_gradient",
      "message": "PGO costs nearly twice multi-objective AdamW and is slightly worse.",
      "severity": 0.5,
      "type": "phase_optimizer_no_frontier_gain"
    }
  ],
  "synthesis": {
    "comparator_warning": "QN-000008 compares complex with two-channel real; later studies compare complex with a cellwise best-real envelope containing an exact real implementation.",
    "final_interpretation": "The evaluated evidence falsifies an intrinsic complex-arithmetic advantage: an exact real block is functionally equivalent and stronger real controls remove the observed robustness gain. The selected quantitative law fails held-out magnitude prediction. QN-GRAND-001 remains unexecuted because mandatory readiness gates failed.",
    "heldout_confirmation": {
      "architecture_effect": {
        "exceeds_practical_threshold": false,
        "mean_complex_minus_best_real": -0.009157985999869804,
        "positive": false
      },
      "best_real_winner_counts": {
        "exact_real_block_operator": 1478,
        "real_polar_operator": 442
      },
      "exact_real_equivalence": {
        "maximum_absolute_ece_difference": 1.1920928955078125e-07,
        "maximum_absolute_nll_difference": 3.5762786865234375e-07,
        "maximum_absolute_top1_difference": 0.0
      },
      "experiment_id": "QN-000042",
      "families": [
        "bayesian_urn",
        "hidden_rule_relational",
        "machine_fault_diagnosis",
        "network_intrusion_reasoning"
      ],
      "hierarchical_bootstrap": {
        "ci_high": -0.004570204104723728,
        "ci_low": -0.013250867711224905,
        "confidence": 0.95,
        "estimate": -0.009157985999869804,
        "generator_families": 4,
        "observations": 640,
        "resamples": 20000,
        "worlds": 128
      },
      "law_confirmation": {
        "all_thresholds_pass": false,
        "confirmation_n": 12,
        "effect_sign_accuracy": 1.0,
        "mean_absolute_error": 0.031264674238761815,
        "r2": -30.936351721073724,
        "threshold_checks": {
          "mae_at_most_maximum": false,
          "r2_at_least_minimum": false,
          "sign_accuracy_at_least_minimum": true
        },
        "thresholds": {
          "maximum_mae": 0.015,
          "minimum_r2": 0.5,
          "minimum_sign_accuracy": 0.8
        }
      },
      "nested_effects": 1920,
      "nested_summary": {
        "mean": -0.009157985999869804,
        "median": 0.0,
        "n": 1920,
        "probability_of_superiority": 0.0,
        "standard_deviation": 0.023548105463035745,
        "trimmed_mean_10pct": -0.0024848088020614036,
        "worst_decile": -0.041666656732559204
      },
      "outcome_eligible": false,
      "positive_nested_effects": 0,
      "training_seeds": 5,
      "world_sign_flip_pvalue_two_sided": 4.9999750001249995e-06,
      "world_summary": {
        "mean": -0.009157985999869804,
        "median": -0.008055555820465087,
        "n": 128,
        "probability_of_superiority": 0.0,
        "standard_deviation": 0.007361363690879325,
        "trimmed_mean_10pct": -0.008397435798094823,
        "worst_decile": -0.019444445073604585
      },
      "worlds": 128,
      "zero_nested_effects": 1538
    },
    "historical_within_neuroworld": {
      "ci_high": 0.06739965130254905,
      "ci_low": 0.05293366263940652,
      "comparator": "two_channel_operator",
      "experiment_id": "QN-000008",
      "moderate_shift_mean_difference": 0.06016665697097778
    },
    "mechanism_discovery": {
      "experiment_id": "QN-000033",
      "interpretation": "Exact real-block computation reproduces complex computation; destructive phase interventions show phase use but not uniquely complex capacity.",
      "outcome_eligible": false
    },
    "outcome_category": "A_falsified_intrinsic_complex_advantage",
    "power_pilot": {
      "estimated_power": 1.0,
      "experiment_id": "QN-000031",
      "outcome_eligible": false,
      "selected_worlds": 32,
      "train_size_1000": {
        "mean": -0.05519245126334431,
        "median": -0.054528985384251064,
        "n": 8,
        "probability_of_superiority": 0.0,
        "standard_deviation": 0.010049786576161244,
        "trimmed_mean_10pct": -0.05519245126334431,
        "worst_decile": -0.06487130236312531
      }
    },
    "qn_grand_001": {
      "blocking_failures": [
        "complete_preregistered_real_envelope",
        "equal_or_real_favoring_search_budgets",
        "compute_matching_records",
        "full_shiftgauntlet_outcome_grid",
        "full_discovery_protocol",
        "raw_predictions_preserved"
      ],
      "executed": false,
      "sealed_benchmark_opened": false,
      "status": "blocked_before_execution"
    },
    "reduced_discovery": {
      "candidate_discovery_mae": 0.0026041648454527134,
      "candidate_discovery_r2": 0.9487095562816686,
      "experiment_id": "QN-000040",
      "frozen_candidate": "quadratic",
      "mean_nested_effect": -0.03695312475028913,
      "nested_effects": 2880,
      "outcome_eligible": false,
      "positive_nested_effects": 0
    },
    "scope": "synthetic and nonclinical computational evidence"
  }
};
