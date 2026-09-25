#[1]: System Parameter
DATATYPE_PRECISION = 32 #Normally set to '32' for seeking. Use '64' only for precision verification against CPU-run simulations.

#[2]: Parameter Test
"""
 * This parameter defines the model to test with a specific set of parameters.
"""
PARAMETERTEST = {'analysisData':           'BSCALPHATEST_ae\\BSCALPHATEST_BTCUSDT',
                 'exitFunctionType':       'BSCALPHA',
                 'balance_initial':        1_000_000,
                 'balance_allocation_max': None,
                 'leverage':               5,
                 'isolated':               True,
                 'tradingFee':             0.0005,
                 'tradeParams':            (1.0, 0.0548),
                 'modelParams':            (0.0, 0.0075, 0.749, 0.0863),
                 'pslReentry':             True,
                }
"""
PARAMETERTEST = {'analysisData':           'USC2_ae\\USC2_BTCUSDT',
                 'exitFunctionType':       'BSCBETA',
                 'balance_initial':        1_000_000,
                 'balance_allocation_max': None,
                 'leverage':               1,
                 'isolated':               False,
                 'tradingFee':             0.0005,
                 'tradeParams':            (0.2662, 0.4793),
                 'modelParams':            (0.0395, 0.0286, 0.0253, 0.5348, 1.0, -0.536, 0.1775, 10.0, 1.9688, 2.6398, 1.3677, 7.4241, 0.0, 5.0316, 3.3866),
                 'pslReentry':             False,
                }

PARAMETERTEST = {'analysisData':           'USC36_ae\\USC36_BTCUSDT',
                 'exitFunctionType':       'BSCALPHA',
                 'balance_initial':        1_000_000,
                 'balance_allocation_max': None,
                 'leverage':               1,
                 'isolated':               False,
                 'tradingFee':             0.0005,
                 'tradeParams':            (0.5325, 0.3594),
                 'modelParams':            (0.0561, 0.0, 0.0349, 0.1875, 0.4187, 0.1056, 0.4624, 0.1079, 0.1288, 0.0415, 0.1929),
                 'pslReentry':             False,
                }
"""

"""
"tradeParams": [
                        0.6009,
                        0.456
                    ],
                    "modelParams": [
                        0.0402,
                        0.0119,
                        0.0165,
                        0.2128,
                        0.5849,
                        0.0442,
                        0.4771,
                        0.0023,
                        0.1526,
                        0.003,
                        0.1676
                    ],
"""



#[3]: Seeker Targets
"""
 * This parameter defines the model 
"""
SEEKERTARGETS = [{'analysisData':               'BSCALPHATRAIN_ae\\BSCALPHATRAIN_BTCUSDT', # Path to the analysis data used for backtesting
                  'exitFunctionType':           'BSCALPHA',                # Type of exit logic model to evaluate
                  'balance_initial':            1_000_000,                 # Initial simulation capital
                  'balance_allocation_max':     None,                      # Maximum capital allowed per trade (None = no limit)
                  'leverage':                   5,                         # Leverage multiplier applied to positions
                  'isolated':                   True,                      # Margin mode (True: Isolated, False: Cross)
                  'tradingFee':                 0.0005,                    # Per-trade fee rate - Edit This Accordingly To The Position Type
                  'pslReentry':                 True,                      # Allow reentry in the same direction after a Position Stop Loss (PSL)
                  'tradeParamConfig':           (None, None),
                  'modelParamConfig':           (None,)*11,
                  'nSeekerPoints':              10_000,        # Number Of Independent Seekers Exploring The Parameter Space Simultaneously
                  'parameterBatchSize':         None,          # GPU Batch Size (None = Auto-Configured)
                  'nRepetition':                100,           # Number of times to repeat the entire exploration lifecycle (epochs/generations)
                  'learningRate':               0.001,         # Base scale of the step size for parameter updates
                  'deltaRatio':                 0.10,          # Perturbation ratio to calculate numerical gradients (e.g., +/- 10% shift)
                  'beta_velocity':              0.999,         # Adam's Beta2 equivalent: EMA decay rate for squared gradients (adjusts step size)
                  'beta_momentum':              0.900,         # Adam's Beta1 equivalent: EMA decay rate for past gradients (adds momentum/inertia)
                  'repopulationRatio':          0.95,          # Proportion of bottom seekers to replace
                  'repopulationInterval':       1,             # Steps between repopulation events
                  'repopulationGuideRatio':     0.05,          # Ratio of new seekers guided by survivors vs completely random
                  'repopulationDecayRate':      0.001,         # How fast the search area narrows down around survivors
                  'scoringSamples':             50,            # Number of recent steps to calculate the EMA of the best score improvements
                  'scoring':                    'SHARPERATIO', # Select From (FINALBALANCE, GROWTHRATE, VOLATILITY, SHARPERATIO)
                  'scoring_maxMDD':             1.0,           # Maximum Drawdown Allowed
                  'scoring_growthRateWeight':   1.0,           # Adjust To Be Somewhere Between 0.0 to 3.0 (If 0.0, Completely Ignored)
                  'scoring_growthRateScaler':   1e6,           # Adjust Such That The Scaled Value Lies Somwhere Between -1.0 to 2.0
                  'scoring_volatilityWeight':   0.20,          # Adjust To Be Somewhere Between 0.0 to 3.0 (If 0.0, Completely Ignored)
                  'scoring_volatilityScaler':   10,            # Adjust Such That The Scaled Value Lies Somwhere Between 0.1 to 3.0
                  'scoring_tradeVolumesWeight': 0.1,           # Adjust To Be Somewhere Between 0.0 to 3.0 (If 0.0, Completely Ignored)
                  'scoring_tradeVolumesScaler': 1e-6,          # Adjust Such That The Scaled Value Lies Somewhere Between 1.0 to 5.0
                  'terminationThreshold':       1e-4,          # If the score improvement EMA falls below this value, terminate the current repetition
                 },
                ]

"""
'paramConfig': [None,   #FSL Immed
                1.0000, #FSL Close
                None,   #Side Offset
                None,   #Theta - SHORT
                None,   #Alpha - SHORT
                None,   #Beta0 - SHORT
                None,   #Beta1 - SHORT
                None,   #Gamma - SHORT
                None,   #Theta - LONG
                None,   #Alpha - LONG
                None,   #Beta0 - LONG
                None,   #Beta1 - LONG
                None    #Gamma - LONG
                ],
"""



#[4]: Result Code to Read
"""
 * This parameter defines the target TEF function optimized parameters search process result to read. The target is the result folder name under the 'results' folder.
 * Example: _RCODETOREAD = 'teffps_result_1768722056'
"""
RCODETOREAD = 'teffps_result_1777435511'



#[5]: Mode
"""
<MODE>
 * 
 * Available Modes: 'TEST'/'SEEK'/'READ'
"""
MODE = 'SEEK'