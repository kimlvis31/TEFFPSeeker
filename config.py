#[1]: System Parameter
DATATYPE_PRECISION = 32 #Normally set to '32' for seeking. Use '64' only for precision verification against CPU-run simulations.

#[2]: Parameter Test
"""
 * This parameter defines the model to test with a specific set of parameters.
"""
PARAMETERTEST = {'analysisData':           'USC0_ae_0\\USC0_0_BTCUSDT',
                 'exitFunctionType':       'MMACDDEFAULT',
                 'balance_initial':        100_000,
                 'balance_allocation_max': 1000,
                 'leverage':               1,
                 'tradeParams':            (0.6626, 0.0),
                 'modelParams':            (3.5880, 1.30, -0.2483, 0.141359, 0.134799, 0.1730),
                 'pslReentry':             False,
                }
"""
PARAMETERTEST = {'analysisData':           'USC0_ae_0\\USC0_0_BTCUSDT',
                 'exitFunctionType':       'SPDDEFAULT',
                 'balance_initial':        100_000,
                 'balance_allocation_max': 1000,
                 'leverage':               1,
                 'tradeParams':            (1.0000, 1.0000),
                 'modelParams':            (-0.4825, 0.173852, 0.741636, -0.3693, 0.12492, 0.715351),
                 'pslReentry':             False,
                }
"""



#[3]: Seeker Targets
"""
 * This parameter defines the model 
"""
SEEKERTARGETS = [{'analysisData':             'USC0_ae_0\\USC0_0_BTCUSDT',
                  'exitFunctionType':         'MMACDDEFAULT',
                  'balance_initial':          100_000,
                  'balance_allocation_max':   1000,
                  'leverage':                 1,
                  'pslReentry':               False,
                  'tradeParamConfig':         (None, None),
                  'modelParamConfig':         (None, None, None, None, None, None),
                  'nSeekerPoints':            1000,
                  'parameterBatchSize':       None,
                  'nRepetition':              5,
                  'learningRate':             0.001,
                  'deltaRatio':               0.10,
                  'beta_velocity':            0.999,
                  'beta_momentum':            0.900,
                  'repopulationRatio':        0.95,
                  'repopulationInterval':     10,
                  'repopulationGuideRatio':   0.5,
                  'repopulationDecayRate':    0.1,
                  'scoringSamples':           50,
                  'scoring':                  'SHARPERATIO',
                  'scoring_maxMDD':           1,
                  'scoring_growthRateWeight': 1.0,
                  'scoring_growthRateScaler': 1e5,
                  'scoring_volatilityWeight': 0.1,
                  'scoring_volatilityScaler': 0.1,
                  'scoring_nTradesWeight':    0.1,
                  'scoring_nTradesScaler':    0.00001,
                  'terminationThreshold':     1e-6,
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
RCODETOREAD = 'teffps_result_1776846287'



#[5]: Mode
"""
<MODE>
 * 
 * Available Modes: 'TEST'/'SEEK'/'READ'
"""
MODE = 'READ'