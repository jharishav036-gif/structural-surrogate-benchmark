# Data dictionary

All datasets follow the same layout: one row per design, `design_id` is a
1-based running index, the next 8-10 columns are the design variables
(features) and the last three columns are the solver-computed response
targets. Units are given in the column names where they are not self-
evident. The solvers work in SI units internally; input columns keep the
engineering units listed below.

## beam_dataset.csv (n = 500)

Cantilever beam with a rectangular section, tip point load P1, mid-span
point load P2 and a uniformly distributed load q. Solved with 40
Euler-Bernoulli beam elements.

| column | unit | range | description |
|---|---|---|---|
| design_id | - | 1-500 | running index |
| L | m | 2.0-6.0 | span |
| b | m | 0.15-0.40 | section width |
| h | m | 0.25-0.70 | section depth |
| E | GPa | 68-210 | Young's modulus |
| rho | kg/m3 | 2650-7900 | density |
| P1 | kN | 5-60 | tip point load |
| P2 | kN | 0-40 | mid-span point load |
| q | kN/m | 0-25 | uniform load |
| tip_deflection_mm | mm | target | deflection at the free end |
| max_stress_MPa | MPa | target | normal stress at the fixed end |
| f1_Hz | Hz | target | first natural frequency |

## frame_dataset.csv (n = 500)

Fixed-base portal frame: two columns (bc x hc) of height H, one beam
(bb x hb) of span S; lateral load W at beam level (split equally between
the joints), uniform gravity load w on the beam. Each member discretised
into 8 frame elements.

| column | unit | range | description |
|---|---|---|---|
| design_id | - | 1-500 | running index |
| H | m | 3.0-7.0 | column height |
| S | m | 4.0-10.0 | beam span |
| bc | m | 0.20-0.45 | column width |
| hc | m | 0.25-0.60 | column depth |
| bb | m | 0.20-0.45 | beam width |
| hb | m | 0.30-0.70 | beam depth |
| E | GPa | 68-210 | Young's modulus |
| rho | kg/m3 | 2650-7900 | density |
| W | kN | 10-120 | lateral load at beam level |
| w | kN/m | 5-60 | uniform load on the beam |
| sway_mm | mm | target | lateral displacement at beam level |
| max_stress_MPa | MPa | target | max normal stress over all members |
| f1_Hz | Hz | target | first natural frequency |

## truss_dataset.csv (n = 500)

6-panel simply supported Pratt truss (statically determinate): span S,
height Ht, member areas by group, mid-span bottom-node point load P and
bottom-chord distributed load w (applied as statically equivalent nodal
loads).

| column | unit | range | description |
|---|---|---|---|
| design_id | - | 1-500 | running index |
| S | m | 8.0-20.0 | span |
| Ht | m | 1.5-4.0 | truss height |
| A_chord | mm2 | 1500-5000 | chord member area (top and bottom) |
| A_diag | mm2 | 800-3000 | diagonal member area |
| A_vert | mm2 | 800-3000 | vertical and end-post member area |
| E | GPa | 68-210 | Young's modulus |
| rho | kg/m3 | 2650-7900 | density |
| P | kN | 20-120 | mid-span bottom-node point load |
| w | kN/m | 5-40 | bottom-chord distributed load |
| mid_deflection_mm | mm | target | deflection of the mid-span bottom node |
| max_stress_MPa | MPa | target | max absolute member normal stress |
| f1_Hz | Hz | target | first natural frequency |

## Physical screening

Sampling is by Latin hypercube over the boxes above. Designs are
retained only if they satisfy the serviceability and strength limits
used in this study:

- beam: tip deflection <= L/100 and max stress <= 350 MPa
- frame: sway <= H/100 and max stress <= 350 MPa
- truss: mid-span deflection <= S/250 and max stress <= 350 MPa

Acceptance rates are recorded in `acceptance_rates.csv`.

## extrapolation/{family}_ext.csv (n = 100 each)

Drawn from boxes extended by 15% on both sides of every parameter range
and passed through the same physical screening. Used to quantify
performance outside the training envelope.

## splits/{family}_folds.csv

| column | description |
|---|---|
| design_id | key into the dataset |
| split | `train` (400 designs) or `test` (100 designs) |
| fold | 0-9 stratified CV fold within the training partition (-1 for test rows) |

The split is common to the three targets of a family. Stratification
uses deciles of the deflection target so that the full response range
is represented in every fold and in the test set.
