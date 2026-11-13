GPU: cpu | total 14.93 h

| stage | count | mean | sum_min |
|---|---|---|---|
| generate | 20 | 998.3 | 332.8 |
| load_data | 19 | 19.8 | 6.3 |
| load_mask_model | 3 | 16.5 | 0.8 |
| load_model | 25 | 43.5 | 18.1 |
| perturb | 3 | 1047.9 | 52.4 |
| score | 40 | 565.3 | 376.8 |
| score_perturbed | 3 | 2168.5 | 108.4 |

| stage | what | seconds |
|---|---|---|
| generate | xsum__Qwen_Qwen2.5-1.5B__seed0__n100__T1.0_k50_p1.0 | 1258.21 |
| score | xsum__Qwen_Qwen2.5-1.5B__seed0__n100__T1.0_k50_p1.0__scorer=Qwen_Qwen2.5-1.5B__auto | 392.76 |
| generate | arwiki__Qwen_Qwen2.5-1.5B__seed0__n100__T1.0_k50_p1.0 | 1682.95 |
| score | arwiki__Qwen_Qwen2.5-1.5B__seed0__n100__T1.0_k50_p1.0__scorer=Qwen_Qwen2.5-1.5B__auto | 846.68 |
| score | arwiki__Qwen_Qwen2.5-1.5B__seed0__n100__T1.0_k50_p1.0__scorer=aubmindlab_aragpt2-medium__auto | 147.44 |
| generate | arwiki__aubmindlab_aragpt2-medium__seed0__n100__T1.0_k50_p1.0 | 546.73 |
| score | arwiki__aubmindlab_aragpt2-medium__seed0__n100__T1.0_k50_p1.0__scorer=Qwen_Qwen2.5-1.5B__auto | 1640.16 |
| score | arwiki__aubmindlab_aragpt2-medium__seed0__n100__T1.0_k50_p1.0__scorer=aubmindlab_aragpt2-medium__auto | 264.97 |
| generate | xsum__gpt2-xl__seed0__n100__T0.7_k50_p1.0 | 578.01 |
| score | xsum__gpt2-xl__seed0__n100__T0.7_k50_p1.0__scorer=gpt2-xl__auto | 408.71 |
| generate | xsum__gpt2-xl__seed0__n100__T1.0_k0_p1.0 | 675.87 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k0_p1.0__scorer=gpt2-xl__auto | 644.16 |
| generate | xsum__gpt2-xl__seed0__n100__T1.0_k0_p0.96 | 603.49 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k0_p0.96__scorer=gpt2-xl__auto | 439.08 |
| generate | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0 | 550.87 |
| score | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto | 441.82 |
| score | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc25 | 74.99 |
| score | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc50 | 144.84 |
| score | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc100 | 287.3 |
| perturb | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__t5-large__k10__n30__pct0.3__span2 | 1218.99 |
| score_perturbed | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__t5-large__k10__n30__pct0.3__span2 | 2609.45 |
| generate | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0 | 903.67 |
| score | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto | 910.8 |
| score | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc25 | 152.16 |
| score | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc50 | 294.9 |
| score | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc100 | 581.54 |
| perturb | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__t5-large__k10__n30__pct0.3__span2 | 906.27 |
| score_perturbed | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__t5-large__k10__n30__pct0.3__span2 | 1587.98 |
| generate | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0 | 916.46 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto | 900.42 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc25 | 167.86 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc50 | 312.29 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc100 | 607.91 |
| perturb | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__t5-large__k10__n30__pct0.3__span2 | 1018.51 |
| score_perturbed | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__t5-large__k10__n30__pct0.3__span2 | 2308.02 |
| generate | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0 | 550.87 |
| score | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto | 441.82 |
| score | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc25 | 74.99 |
| score | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc50 | 144.84 |
| score | squad__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc100 | 287.3 |
| generate | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0 | 903.67 |
| score | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto | 910.8 |
| score | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc25 | 152.16 |
| score | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc50 | 294.9 |
| score | writing__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc100 | 581.54 |
| generate | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0 | 916.46 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto | 900.42 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc25 | 167.86 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc50 | 312.29 |
| score | xsum__gpt2-xl__seed0__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto__trunc100 | 607.91 |
| generate | squad__EleutherAI_gpt-neo-2.7B__seed0__n100__T1.0_k50_p1.0 | 1405.47 |
| score | squad__EleutherAI_gpt-neo-2.7B__seed0__n100__T1.0_k50_p1.0__scorer=EleutherAI_gpt-neo-2.7B__auto | 1311.31 |
| generate | writing__EleutherAI_gpt-neo-2.7B__seed0__n100__T1.0_k50_p1.0 | 1387.57 |
| score | writing__EleutherAI_gpt-neo-2.7B__seed0__n100__T1.0_k50_p1.0__scorer=EleutherAI_gpt-neo-2.7B__auto | 1266.38 |
| generate | xsum__EleutherAI_gpt-neo-2.7B__seed0__n100__T1.0_k50_p1.0 | 1535.52 |
| score | xsum__EleutherAI_gpt-neo-2.7B__seed0__n100__T1.0_k50_p1.0__scorer=EleutherAI_gpt-neo-2.7B__auto | 1295.59 |
| generate | squad__facebook_opt-2.7b__seed0__n100__T1.0_k50_p1.0 | 1724.46 |
| score | squad__facebook_opt-2.7b__seed0__n100__T1.0_k50_p1.0__scorer=facebook_opt-2.7b__auto | 1338.53 |
| generate | writing__facebook_opt-2.7b__seed0__n100__T1.0_k50_p1.0 | 1183.77 |
| score | writing__facebook_opt-2.7b__seed0__n100__T1.0_k50_p1.0__scorer=facebook_opt-2.7b__auto | 751.87 |
| generate | xsum__facebook_opt-2.7b__seed0__n100__T1.0_k50_p1.0 | 1286.35 |
| score | xsum__facebook_opt-2.7b__seed0__n100__T1.0_k50_p1.0__scorer=facebook_opt-2.7b__auto | 975.55 |
| generate | xsum__gpt2-xl__seed1__n100__T1.0_k50_p1.0 | 611.37 |
| score | xsum__gpt2-xl__seed1__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto | 431.69 |
| generate | xsum__gpt2-xl__seed2__n100__T1.0_k50_p1.0 | 744.27 |
| score | xsum__gpt2-xl__seed2__n100__T1.0_k50_p1.0__scorer=gpt2-xl__auto | 702.04 |
