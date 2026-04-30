#[1]: System Parameter
DATATYPE_PRECISION = 32 #Normally set to '32' for seeking. Use '64' only for precision verification against CPU-run simulations.

#[2]: Parameter Test
"""
 * This parameter defines the model to test with a specific set of parameters.
"""
PARAMETERTEST = {'analysisData':           'USC38_ae\\USC38_BTCUSDT',
                 'exitFunctionType':       'BSCBETA',
                 'balance_initial':        1_000_000,
                 'balance_allocation_max': None,
                 'leverage':               1,
                 'isolated':               False,
                 'tradeParams':            (0.9152, 0.5259),
                 'modelParams':            (0.0508, 0.0389, 0.0353, 0.3568, 1.0, -0.2752, 0.2132, 9.9876, 2.3638, 1.6455, 1.8275, 7.7177, 7.0058, 1.6737, 9.0432),
                 'pslReentry':             False,
                }
"""
PARAMETERTEST = {'analysisData':           'USC36_ae\\USC36_BTCUSDT',
                 'exitFunctionType':       'BSCALPHA',
                 'balance_initial':        1_000_000,
                 'balance_allocation_max': None,
                 'leverage':               1,
                 'isolated':               False,
                 'tradeParams':            (0.5325, 0.3594),
                 'modelParams':            (0.0561, 0.0, 0.0349, 0.1875, 0.4187, 0.1056, 0.4624, 0.1079, 0.1288, 0.0415, 0.1929),
                 'pslReentry':             False,
                }
"""



#[3]: Seeker Targets
"""
 * This parameter defines the model 
"""
SEEKERTARGETS = [{'analysisData':               'USC38_ae\\USC38_BTCUSDT',
                  'exitFunctionType':           'BSCBETA',
                  'balance_initial':            1_000_000,
                  'balance_allocation_max':     None,
                  'leverage':                   1,
                  'isolated':                   True,
                  'pslReentry':                 False,
                  'tradeParamConfig':           (None, None),
                  'modelParamConfig':           (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
                  'nSeekerPoints':              1_000,
                  'parameterBatchSize':         None,
                  'nRepetition':                100,
                  'learningRate':               0.001,
                  'deltaRatio':                 0.10,
                  'beta_velocity':              0.999,
                  'beta_momentum':              0.900,
                  'repopulationRatio':          0.95,          # Proportion of bottom seekers to replace
                  'repopulationInterval':       3,             # Steps between repopulation events
                  'repopulationGuideRatio':     0.1,           # Ratio of new seekers guided by survivors vs completely random
                  'repopulationDecayRate':      0.05,          # How fast the search area narrows down around survivors
                  'scoringSamples':             50,            # Scoring Moving Average Number Of Samples
                  'scoring':                    'SHARPERATIO', # Select From (FINALBALANCE, GROWTHRATE, VOLATILITY, SHARPERATIO)
                  'scoring_maxMDD':             1,
                  'scoring_growthRateWeight':   0.8,
                  'scoring_growthRateScaler':   1e7,
                  'scoring_volatilityWeight':   1.0,
                  'scoring_volatilityScaler':   10,
                  'scoring_tradeVolumesWeight': 1.00,
                  'scoring_tradeVolumesScaler': 0.00001,
                  'terminationThreshold':       1e-6,
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