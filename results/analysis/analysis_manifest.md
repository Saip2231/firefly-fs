# Analysis manifest (Stage A9)

- Date: 2026-10-31T00:07:22
- git: 1217e0d+dirty
- Command (all analyze stages): `python -m experiments.analyze`

## A0

- Command: `python -m experiments.analyze --only A0`
- Sources:

| source | rows | sha256 |
|---|---|---|
| `results/baselines_digits.csv` | 190 | `c18128300b68d2ff5c1918dab2d54163bd80eee64b6a46b9d0708a0bc433de59` |
| `results/baselines_cifar_r18.csv` | 250 | `d057cb38f60027ce27ffe2f7460a99836e6c2bdb01a0abbe91405e43946629b7` |
| `results/meta_digits.csv` | 320 | `c60bc1978ef632604550edcfad6e2d473df6a3d030c38b06b53ac013eeee4728` |
| `results/meta_cifar_r18.csv` | 640 | `7c3a90ed3d4dcbbff56f3b041b9818e04d9dccd9d137616eaccd5765dd89a3b2` |
| `results/split_hashes.csv` | 20 | `70adbcb285d85ba33e68adc0bad41c4ce87106ff7071eeb9e773129898c6c6bd` |
| `PREREG.md` | - | `92b6b0d35957324de51c3fce93df6cd62f2d4285751199aeff87972bdd3bd987` |

| output | bytes | sha256 |
|---|---|---|
| `results/analysis/integrity_report.md` | 2319 | `71278c421977cb6efe96852ac434efee8cb86f63cae39ffd6fa5bd151ca723cc` |

## A1

- Command: `python -m experiments.analyze --only A1`
- Sources:

| source | rows | sha256 |
|---|---|---|
| `results/baselines_digits.csv` | 190 | `c18128300b68d2ff5c1918dab2d54163bd80eee64b6a46b9d0708a0bc433de59` |
| `results/baselines_cifar_r18.csv` | 250 | `d057cb38f60027ce27ffe2f7460a99836e6c2bdb01a0abbe91405e43946629b7` |
| `results/meta_digits.csv` | 320 | `c60bc1978ef632604550edcfad6e2d473df6a3d030c38b06b53ac013eeee4728` |
| `results/meta_cifar_r18.csv` | 640 | `7c3a90ed3d4dcbbff56f3b041b9818e04d9dccd9d137616eaccd5765dd89a3b2` |
| `results/split_hashes.csv` | 20 | `70adbcb285d85ba33e68adc0bad41c4ce87106ff7071eeb9e773129898c6c6bd` |
| `PREREG.md` | - | `92b6b0d35957324de51c3fce93df6cd62f2d4285751199aeff87972bdd3bd987` |

| output | bytes | sha256 |
|---|---|---|
| `results/analysis/rq1_tests.csv` | 8575 | `37eb2d4397ff28aec28f81d6a0c5090b9be754f06646af077edcdaf334877ff5` |
| `results/analysis/descriptive.csv` | 10139 | `43137c7fa990e18a64e806cf924b61293794543c55ddd47345d84a72d15815bc` |
| `results/analysis/rq1_summary.md` | 2420 | `351058f30e06bebbda60fa3fa36b482f092f047cca133f99e73a42fbbc17e35c` |

## A2

- Command: `python -m experiments.analyze --only A2`
- Sources:

| source | rows | sha256 |
|---|---|---|
| `results/baselines_digits.csv` | 190 | `c18128300b68d2ff5c1918dab2d54163bd80eee64b6a46b9d0708a0bc433de59` |
| `results/baselines_cifar_r18.csv` | 250 | `d057cb38f60027ce27ffe2f7460a99836e6c2bdb01a0abbe91405e43946629b7` |
| `results/meta_digits.csv` | 320 | `c60bc1978ef632604550edcfad6e2d473df6a3d030c38b06b53ac013eeee4728` |
| `results/meta_cifar_r18.csv` | 640 | `7c3a90ed3d4dcbbff56f3b041b9818e04d9dccd9d137616eaccd5765dd89a3b2` |
| `results/split_hashes.csv` | 20 | `70adbcb285d85ba33e68adc0bad41c4ce87106ff7071eeb9e773129898c6c6bd` |

| output | bytes | sha256 |
|---|---|---|
| `figures/F1_acc_vs_k_cifar_r18_b3000.pdf` | 25251 | `dda2a35916cc1bf464402516205aeb2126ac8b08ba341fa539f4b5bde3aa8867` |
| `figures/F1_acc_vs_k_cifar_r18_b3000.png` | 205931 | `c2e0412631bdb719121f491688aef8c9329f6e7b6b3509750e4de874c7a4c83d` |
| `figures/F1_acc_vs_k_cifar_r18_b800.pdf` | 25256 | `3640c3fb3fc1843cb8d8e7c66cdcc688bfbd9ea71b22f0caa3de8b032d0fa017` |
| `figures/F1_acc_vs_k_cifar_r18_b800.png` | 204869 | `0f91bc000ff6035fe33b6c858e24d8611f4eb26882b0ca30ef9ca3ba896faf00` |
| `figures/F1_acc_vs_k_digits_b3000.pdf` | 24037 | `2e99a7f9c75f9a14a68b3d049f1e40b497c704c8b29fdc6f99ff44e5f53d0adc` |
| `figures/F1_acc_vs_k_digits_b3000.png` | 133682 | `557910e05dfc7a865b4753321ee895260c21b6fbce4f3c39d0cdfc1efb1f6a8c` |
| `figures/F1_acc_vs_k_digits_b800.pdf` | 24040 | `ae2c64d529f56fdba59fd280586c1a985475f1ef5a46e8f7d905c600735766f8` |
| `figures/F1_acc_vs_k_digits_b800.png` | 132965 | `8d08a786d9eb7e9cda9070880930ed86577c2a72bf06a8cd4a7a29805ee86d39` |
| `figures/F2_rq1_grid.pdf` | 25168 | `62b40035107a797c9c2ab77111b75a7bc842273dfc87f4c2d789b3aeedd64748` |
| `figures/F2_rq1_grid.png` | 225471 | `789f84d54426e91c2524dcd3492b54725b201c995cf64c1e26a4b0990f65e29d` |
| `figures/data/F1_frames.csv` | 5966 | `c84796b2ab01faeea583754aec7ea9cdff0a46a0ffa3389f787fca7984002e25` |

## A3

- Command: `python -m experiments.analyze --only A3`
- Sources:

| source | rows | sha256 |
|---|---|---|
| `results/baselines_digits.csv` | 190 | `c18128300b68d2ff5c1918dab2d54163bd80eee64b6a46b9d0708a0bc433de59` |
| `results/baselines_cifar_r18.csv` | 250 | `d057cb38f60027ce27ffe2f7460a99836e6c2bdb01a0abbe91405e43946629b7` |
| `results/meta_digits.csv` | 320 | `c60bc1978ef632604550edcfad6e2d473df6a3d030c38b06b53ac013eeee4728` |
| `results/meta_cifar_r18.csv` | 640 | `7c3a90ed3d4dcbbff56f3b041b9818e04d9dccd9d137616eaccd5765dd89a3b2` |

| output | bytes | sha256 |
|---|---|---|
| `results/analysis/rq2_cost.csv` | 79433 | `29b7fdff158166b6e6f3f976d02d2514dec788df6d2b15226aee1d9e5ef35596` |
| `figures/F4_acc_vs_search_cifar_r18.pdf` | 37711 | `af46b4e591cf80a97c00863c8f42fddb3c4d6b1cd976e56c052e107ccd2d363b` |
| `figures/F4_acc_vs_search_cifar_r18.png` | 224963 | `ac2272fbe9000f8ee444e538b473d596e9c01ca501327632b4a3b10dbfdc4a9c` |
| `figures/F4_acc_vs_search_digits.pdf` | 30608 | `4f79020495b9b85d79761781c1eb24d16b08f651ca4ac76372e2d67654e618bc` |
| `figures/F4_acc_vs_search_digits.png` | 162614 | `ed2ef7b148b9824a4a8770e9ec43ec5845beeefe3b7c6125461fe7ac20273904` |
| `figures/F5_break_even_cifar_r18.pdf` | 26148 | `daed630224877e2b7d824a9390ed2864341d6a8eb2110d51720abc1f54f6c7c0` |
| `figures/F5_break_even_cifar_r18.png` | 136708 | `62301a6a4c7e53b86397aa9ea1e0c7aab604e608a7bd9886e20f5f42d0e0edf9` |
| `figures/F5_break_even_digits.pdf` | 24653 | `02654372e4121d4ba37db9afa3b4409aa893ad55ab8b532b4f4bfa61daa9195b` |
| `figures/F5_break_even_digits.png` | 161482 | `989f5b17a7340993230398fa71dfc0923de2b12395e923189e0b7887c861305f` |

## A4

- Command: `python experiments/time_final.py  (idle machine; not rerun by analyze)`
- Sources:

| source | rows | sha256 |
|---|---|---|
| `results/baselines_digits.csv` | 190 | `c18128300b68d2ff5c1918dab2d54163bd80eee64b6a46b9d0708a0bc433de59` |
| `results/baselines_cifar_r18.csv` | 250 | `d057cb38f60027ce27ffe2f7460a99836e6c2bdb01a0abbe91405e43946629b7` |
| `results/meta_digits.csv` | 320 | `c60bc1978ef632604550edcfad6e2d473df6a3d030c38b06b53ac013eeee4728` |
| `results/meta_cifar_r18.csv` | 640 | `7c3a90ed3d4dcbbff56f3b041b9818e04d9dccd9d137616eaccd5765dd89a3b2` |
| `config/frozen_config.yaml` | - | `b14f6b194e4f7b9db64a0ad4ff20bb681dd1a33d9ee7307ee8506e35e614569d` |

| output | bytes | sha256 |
|---|---|---|
| `results/analysis/timing_idle.csv` | 26237 | `cc04f19c533863b38ca0b3f0fd0ce59c8da558fa0158e42a7d719522ea96ae27` |
| `results/analysis/contention.csv` | 788 | `01536d32fc81f17b1038d864b3e468469aadebaf7eec089a0c27eb5068f2af2d` |
| `results/analysis/contention_factor.md` | 226 | `f0e763b38d2eaa79bc647c9fcbe2d78c0f2c102fda4b05d3bc215d9cd67d08f6` |

## A5

- Command: `python -m experiments.analyze --only A5`
- Sources:

| source | rows | sha256 |
|---|---|---|
| `results/baselines_digits.csv` | 190 | `c18128300b68d2ff5c1918dab2d54163bd80eee64b6a46b9d0708a0bc433de59` |
| `results/baselines_cifar_r18.csv` | 250 | `d057cb38f60027ce27ffe2f7460a99836e6c2bdb01a0abbe91405e43946629b7` |
| `results/meta_digits.csv` | 320 | `c60bc1978ef632604550edcfad6e2d473df6a3d030c38b06b53ac013eeee4728` |
| `results/meta_cifar_r18.csv` | 640 | `7c3a90ed3d4dcbbff56f3b041b9818e04d9dccd9d137616eaccd5765dd89a3b2` |

| output | bytes | sha256 |
|---|---|---|
| `results/analysis/rq3_stability.csv` | 43083 | `1ec84559435b387ce01700f136a8d6c054c94af3b5bf028a97cf77b1025e0198` |
| `results/analysis/rq3_feature_frequency.csv` | 3932 | `dde03928488e24fbb2255317f41552142593c39776509b435021e5460986a1ac` |
| `figures/F6_stability_cifar_r18.pdf` | 29276 | `518f402030be12c285e0374d1415b1cb40cf030767961bd66e0cc1b19b39e9fa` |
| `figures/F6_stability_cifar_r18.png` | 201258 | `40f3ad47e25e5d986c9aeebf01870328dc41c7107fdecd6428e5126bde76fcf4` |
| `figures/F6_stability_digits.pdf` | 25443 | `def38acb8f50afed9c9709f5ea7013d8f96b940932a8d4d103d4a12b2abc096e` |
| `figures/F6_stability_digits.png` | 128397 | `068ba5854d94e87e7f9ffcac36c804eb003c583d68c5c11a38c69ade942a050a` |
| `figures/F7_feature_frequency_cifar_r18_k128.pdf` | 21915 | `1b78b7fe8ff37011c81e0f7ecde35b174299fbd30fd890ad30f0dabf22d7ac14` |
| `figures/F7_feature_frequency_cifar_r18_k128.png` | 97117 | `6ccec4059ab413292fc06f54029e48a739ca49d7d681290f684ca0870b02e9e9` |
| `figures/F7_feature_frequency_cifar_r18_k16.pdf` | 21615 | `6c1c78b6331e54a342208d0c2b015b20b18e423e7299a73660fa0c5d18da8d1a` |
| `figures/F7_feature_frequency_cifar_r18_k16.png` | 95461 | `dbf77cddeda17921a5b6b3f29fe5375ded49f8ba44755c2fd21497e9998b5eef` |
| `figures/F7_feature_frequency_cifar_r18_k32.pdf` | 21728 | `8262b24215244ec46e38b61eb9ebc3d1592d10ee354d37e4663f1068387c41b6` |
| `figures/F7_feature_frequency_cifar_r18_k32.png` | 96356 | `dc3f5b96b48c873f7a38b73c9bdbe231f75a8b5dadec340c8694a2e40a383f8a` |
| `figures/F7_feature_frequency_cifar_r18_k64.pdf` | 21766 | `93afb8201b75f6ede0104f5402876bd3cfe5642aa2ce7dfb3559ef0d2c7a9bf0` |
| `figures/F7_feature_frequency_cifar_r18_k64.png` | 96250 | `63deb9ee631a2a751a27af1ca56557b0ed4377e8a6912af8274ced2ca85e2100` |
| `figures/F7_feature_frequency_digits_k16.pdf` | 22152 | `7e98e011a5201dbf2dbdf57e893f4fe689d53eae4bf92b9e573173297b720bbc` |
| `figures/F7_feature_frequency_digits_k16.png` | 92014 | `ddc795772357cb9c57548ef873111de029682e7a7e36dbb5e96a8ff99ecde7f7` |
| `figures/F7_feature_frequency_digits_k32.pdf` | 22256 | `292591f811825dfb5505f59b3dcea485e88a9efe33901883c809ba5eef32ee30` |
| `figures/F7_feature_frequency_digits_k32.png` | 91545 | `4e2e95f82db6690662cf8946c45a8e0ae1067c2bcd3804006325490c6380db42` |

## A6a

- Command: `python experiments/eval_baseline_transfer.py  (touches the test split via the runner's evaluate_on_test; not rerun by analyze)`
- Sources:

| source | rows | sha256 |
|---|---|---|
| `results/baselines_digits.csv` | 190 | `c18128300b68d2ff5c1918dab2d54163bd80eee64b6a46b9d0708a0bc433de59` |
| `results/baselines_cifar_r18.csv` | 250 | `d057cb38f60027ce27ffe2f7460a99836e6c2bdb01a0abbe91405e43946629b7` |

| output | bytes | sha256 |
|---|---|---|
| `results/baselines_transfer_digits.csv` | 14748 | `e5e4f0226e1052366598efc219d3e09f6d8733a4d638610c579179292b04b04e` |
| `results/baselines_transfer_cifar_r18.csv` | 54942 | `c1caf38b966ef8425f21e8ff16fcbaf50020fd927cbd11e3fef3f1690422b194` |

## A6

- Command: `python -m experiments.analyze --only A6`
- Sources:

| source | rows | sha256 |
|---|---|---|
| `results/baselines_transfer_digits.csv` | 100 | `e5e4f0226e1052366598efc219d3e09f6d8733a4d638610c579179292b04b04e` |
| `results/baselines_transfer_cifar_r18.csv` | 200 | `c1caf38b966ef8425f21e8ff16fcbaf50020fd927cbd11e3fef3f1690422b194` |
| `results/meta_digits.csv` | 320 | `c60bc1978ef632604550edcfad6e2d473df6a3d030c38b06b53ac013eeee4728` |
| `results/meta_cifar_r18.csv` | 640 | `7c3a90ed3d4dcbbff56f3b041b9818e04d9dccd9d137616eaccd5765dd89a3b2` |

| output | bytes | sha256 |
|---|---|---|
| `results/analysis/rq4_transfer.csv` | 81468 | `eb5b9a1040d0b02a6228880fcbe6c58eb8c5e25e0bed1ee219a6b17f69918153` |
| `results/analysis/rq4_gaps.csv` | 22714 | `a79f22a507e1e5856d1c233a3f9d68a8b9e692609f25b4ae9c4d77819b083295` |
| `figures/F8_transfer_cifar_r18.pdf` | 30791 | `e5b28b4a3325750e8c09ae41e927418dfedad9274de901b046ac756956c2172d` |
| `figures/F8_transfer_cifar_r18.png` | 307854 | `0ae78755f4b5306680e53b705589dd880f204c4fa9420a02bfa23829713d236d` |
| `figures/F8_transfer_digits.pdf` | 28064 | `542817de18d5482a297d4e858dbd7f5a85f61b98fe2359c6ca7251a8482f4eba` |
| `figures/F8_transfer_digits.png` | 212178 | `deb4efe4652579cfb237d829c9856ed7659311fcfd562ad19c9be9d76b637693` |

## A7

- Command: `python -m experiments.analyze --only A7`
- Sources:

| source | rows | sha256 |
|---|---|---|
| `results/baselines_digits.csv` | 190 | `c18128300b68d2ff5c1918dab2d54163bd80eee64b6a46b9d0708a0bc433de59` |
| `results/baselines_cifar_r18.csv` | 250 | `d057cb38f60027ce27ffe2f7460a99836e6c2bdb01a0abbe91405e43946629b7` |
| `results/meta_digits.csv` | 320 | `c60bc1978ef632604550edcfad6e2d473df6a3d030c38b06b53ac013eeee4728` |
| `results/meta_cifar_r18.csv` | 640 | `7c3a90ed3d4dcbbff56f3b041b9818e04d9dccd9d137616eaccd5765dd89a3b2` |

| output | bytes | sha256 |
|---|---|---|
| `results/analysis/ablations.csv` | 23109 | `4a4802bcee7cd335cca90dd9745b215188169db4036ea4530240b346b6ffb452` |
| `figures/F3_budget_curve_cifar_r18.pdf` | 29295 | `409e5d41443ed156300bf5ced508b945803a88016d311bbfc408a6f352283647` |
| `figures/F3_budget_curve_cifar_r18.png` | 221354 | `ad6b290394e51a7eb47dab6c0aad062e81b957213cc26e32f380c69d8303526a` |
| `figures/F3_budget_curve_digits.pdf` | 24934 | `1b2effe9068dba127e2709c344dc78a9011afe653c20a623152ab96f6315ebda` |
| `figures/F3_budget_curve_digits.png` | 139182 | `a2b1efa25e26aa2a0b6c82684213dea9aabff8f6dfc830eeaa97005f20eb8889` |
| `figures/F9_ablations_cifar_r18.pdf` | 25300 | `9d7ab9b2f5adeab4a15fd0560f9c767a937269bfc8edbf7e453901d1ec1ea165` |
| `figures/F9_ablations_cifar_r18.png` | 119943 | `85037b35be5338fb287e10c1f8dc67e3837d2ff245fbe23bc1071096b41d43aa` |
| `figures/F9_ablations_digits.pdf` | 22354 | `5b59d2cfad65921d82e4b1ce7d3e6b7adf614d7b96462f8850b6da00e643ffe3` |
| `figures/F9_ablations_digits.png` | 146837 | `17bd3cbd1a16c94254eb455119c292784d0995d384db2e399613c63d4c63f6d2` |

## A8

- Command: `python -m experiments.analyze --only A8  (skipped while no traces exist)`
- Note: A8 skipped: no results\traces/*.npz traces (deferred; convergence figure not produced)

| output | bytes | sha256 |
|---|---|---|
| (none) | | |

## shared

- Command: `python -m experiments.analyze  (rows upserted by A2, A3, A5, A6, A7)`

| output | bytes | sha256 |
|---|---|---|
| `figures/takeaways.csv` | 9766 | `acdf4a68313ce4b794c8b904849f3c2dcb67292954f8351afc10987d006b7405` |
