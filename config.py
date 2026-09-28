#[1]: System Parameter
DATATYPE_PRECISION = 64 #Normally set to '32' for seeking. Use '64' only for precision verification against CPU-run simulations.

#[2]: Parameter Test
"""
 * This parameter defines the model to test with a specific set of parameters.
"""
PARAMETERTEST = {'analysisData':           'USC0_ae\\USC0_XRPUSDT',
                 'exitFunctionType':       'PATHFINDER',
                 'balance_initial':        100_000,
                 'balance_allocation_max': None,
                 'leverage':               1,
                 'isolated':               True,
                 'orderType':              'ADAPTIVE', # Order type (LIMIT / MARKET / ADAPTIVE)
                 'orderOffset':            0.0005,     # Limit order price offset from the close price
                 'tradingFee_limit':       0.0002,     # Maker fee rate (applied to limit fills)
                 'tradingFee_market':      0.0005,     # Taker fee rate (applied to market fills)
                 'tradeParams':            (1.0, 1.0),
                 'modelParams':            (0.3853, -0.2617, 0.7743, 0.6157, 0.9195, 0.9268, 0.1969, 0.5421, 0.2778, 0.0105, 0.6605, 0.6527, 0.2036, 0.0105, 1.9909, 1.7533, 0.283, 0.9995, 0.9867, 1.2591, 0.9268, 0.981, 0.0758, 0.0811),
                 'pslReentry':             True,
                }



#[3]: Seeker Targets
"""
 * This parameter defines the seeker configuration.
"""
SEEKERTARGETS = [{'analysisData':               'USC0_ae\\USC0_XRPUSDT', # Path to the analysis data used for backtesting
                  'exitFunctionType':           'PATHFINDER',            # Type of exit logic model to evaluate
                  'balance_initial':            10_000,                  # Initial simulation capital
                  'balance_allocation_max':     None,                    # Maximum capital allowed per trade (None = no limit)
                  'leverage':                   1,                       # Leverage multiplier applied to positions
                  'isolated':                   True,                    # Margin mode (True: Isolated, False: Cross)
                  'orderType':                  'ADAPTIVE',              # Order type (LIMIT / MARKET / ADAPTIVE)
                  'orderOffset':                0.0005,                  # Limit order price offset from the close price
                  'tradingFee_limit':           0.0002,                  # Maker fee rate (applied to limit fills)
                  'tradingFee_market':          0.0005,                  # Taker fee rate (applied to market fills)
                  'pslReentry':                 True,                    # Allow reentry in the same direction after a Position Stop Loss (PSL)
                  'tradeParamConfig':           (1.0000, 1.0000),
                  'modelParamConfig':           (None,)*24,
                  'nSeekerPoints':              1_000,         # Number Of Independent Seekers Exploring The Parameter Space Simultaneously
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
                  'scoring_volatilityWeight':   0.30,          # Adjust To Be Somewhere Between 0.0 to 3.0 (If 0.0, Completely Ignored)
                  'scoring_volatilityScaler':   10,            # Adjust Such That The Scaled Value Lies Somwhere Between 0.1 to 3.0
                  'scoring_tradeVolumesWeight': 0.2,           # Adjust To Be Somewhere Between 0.0 to 3.0 (If 0.0, Completely Ignored)
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
MODE = 'TEST'