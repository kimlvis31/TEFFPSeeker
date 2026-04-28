import triton
import triton.language as tl
from triton.language.extra import libdevice as tl_ld
from . import simulatorFunctions as sf

"""
FUNCTION MODEL: MMACDDEFAULT (MMACD Default)
 * The first two parameters are required by the system, and must always be included in the format as they are.
"""
MODEL = [{'PRECISION': 4, 'LIMIT': (0.0001,   5.0000)},   #Alpha
         {'PRECISION': 2, 'LIMIT': (1.00,     10.00)},    #Beta
         {'PRECISION': 4, 'LIMIT': (-1.0000,  1.0000)},   #Delta
         {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Strength - SHORT
         {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Strength - LONG
         {'PRECISION': 4, 'LIMIT': (0.0000,   1.0000)},   #Downhill Threshold
        ]

INPUTDATAKEYS = ['0_MMACD_MSDELTAABSMAREL',]

def PROCESSBATCH(**kwargs):
    sf.processBatch(tkf = processBatch, **kwargs)

"""
<Triton Kernel Function>
 * This is an TEF value calculation function written in Triton.
 * It simply takes in model parameters, model state trackers, and base data, and calculate TEF value for trading simulation in the base Triton Kernel Function.
 * This is an example and is recommended to be kept without edits for reference. The user may add similar .py files following the general structure in this file to test their customized strategies. In order for the trade simulator function to be able to 
   recognize and call this function, the user must implement the model parameter import, state trackers initialization, and function call parts for the new specific model. Check 'processBatch_triton_kernel' function in 'exitFunction_base.py'
"""
#Batch Processing Triton Kernel Function ================================================================================================================================================================================================================
@triton.autotune(configs=sf.TRITON_AUTOTUNE_CONFIGURATIONS, key=sf.TRITON_AUTOTUNE_KEY)
@triton.jit
def processBatch(
    #Constants
    balance_initial:        tl.constexpr,
    balance_allocation_max: tl.constexpr,
    step_price:             tl.constexpr,
    step_quantity:          tl.constexpr,
    step_quote:             tl.constexpr,
    leverage:               tl.constexpr,
    isolated:               tl.constexpr,
    allocationRatio:        tl.constexpr,
    tradingFee:             tl.constexpr,
    lmTable,
    lmTable_stride:         tl.constexpr,
    lmTable_nTiers:         tl.constexpr,
    #Base Data
    data_prices,
    data_prices_stride: tl.constexpr,
    data_analysis,
    data_analysis_stride: tl.constexpr,
    params_trade_fslImmed,
    params_trade_fslClose,
    params_trade_pslReentry: tl.constexpr,
    params_model,
    params_model_stride: tl.constexpr,
    #Result Buffers
    balance_finals,
    balance_bestFit_intercepts,
    balance_bestFit_growthRates,
    balance_bestFit_volatilities,
    balance_wallet_history,
    balance_margin_history,
    balance_ftIndexes,
    nTrades_rb,
    #Sizes
    size_paramsBatch: tl.constexpr,
    size_dataLen:     tl.constexpr,
    size_block:       tl.constexpr,
    #Mode
    SEEKERMODE: tl.constexpr
    ):
    (offsets,
     mask,
     tp_fsl_immed,
     tp_fsl_close,
     balance_cross,
     balance_isolated,
     balance_wallet,
     balance_margin,
     balance_allocated,
     balance_ftIndex,
     quantity,
     entryPrice,
     forceExited,
     nTrades,
     bt_sum,
     bt_sum_xy,
     bt_sum_squared
    ) = sf.initializeSimulation_triton_kernel(
        balance_initial        = balance_initial,
        balance_allocation_max = balance_allocation_max,
        params_trade_fslImmed  = params_trade_fslImmed,
        params_trade_fslClose  = params_trade_fslClose,
        allocationRatio        = allocationRatio,
        size_paramsBatch       = size_paramsBatch, 
        size_block             = size_block
        )

    #Model Parameters
    mp_base_ptr = params_model + (offsets * params_model_stride)
    mp_alpha       = tl.load(mp_base_ptr + 0, mask = mask)
    mp_beta        = tl.load(mp_base_ptr + 1, mask = mask)
    mp_delta       = tl.load(mp_base_ptr + 2, mask = mask)
    mp_strength_S  = tl.load(mp_base_ptr + 3, mask = mask)
    mp_strength_L  = tl.load(mp_base_ptr + 4, mask = mask)
    mp_dhThreshold = tl.load(mp_base_ptr + 5, mask = mask)
    
    #Model State Trackers
    st_tefVal_prev                = tl.full([size_block,], 0.0, dtype=sf.DTYPE)
    st_tefVal_absMax_prev         = tl.full([size_block,], 0.0, dtype=sf.DTYPE)
    st_mmacd_msDeltaAbsMARel_prev = tl.full([],            0.0, dtype=sf.DTYPE)
    st_isDownhill_prev            = tl.full([size_block,], 0.0, dtype=sf.DTYPE)

    #Loop
    for loop_index in range(0, size_dataLen):
        #[1]: TEF Values  <!!! EDIT HERE FOR MODEL ADDITION !!!> --------------------------------------------------------------------------------------------------------------------------------------
        (tefDir_this,
         tefVal_this, 
         st_tefVal_prev,
         st_tefVal_absMax_prev,
         st_mmacd_msDeltaAbsMARel_prev,
         st_isDownhill_prev
        ) = getTEFValue(
            #Process
            size_block = size_block,
            #Base Data
            loop_index           = loop_index,
            data_analysis        = data_analysis, 
            data_analysis_stride = data_analysis_stride,
            #Model Parameters
            mp_alpha       = mp_alpha,
            mp_beta        = mp_beta,
            mp_delta       = mp_delta,
            mp_strength_S  = mp_strength_S,
            mp_strength_L  = mp_strength_L,
            mp_dhThreshold = mp_dhThreshold,
            #Model State Trackers
            st_tefVal_prev                = st_tefVal_prev,
            st_tefVal_absMax_prev         = st_tefVal_absMax_prev,
            st_mmacd_msDeltaAbsMARel_prev = st_mmacd_msDeltaAbsMARel_prev,
            st_isDownhill_prev            = st_isDownhill_prev
            )
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        #[2]: Trade Simulation
        (balance_cross,
         balance_isolated,
         balance_wallet,
         balance_margin,
         balance_allocated,
         balance_ftIndex,
         quantity, 
         entryPrice, 
         forceExited, 
         nTrades,
         bt_sum, 
         bt_sum_xy,
         bt_sum_squared
        ) = sf.processTrade_triton_kernel(
            #Process
            offsets      = offsets, 
            mask         = mask, 
            size_dataLen = size_dataLen,
            SEEKERMODE   = SEEKERMODE,
            #Constants
            balance_initial        = balance_initial,
            balance_allocation_max = balance_allocation_max,
            step_price             = step_price,
            step_quantity          = step_quantity,
            step_quote             = step_quote,
            leverage               = leverage,
            isolated               = isolated,
            allocationRatio        = allocationRatio,
            tradingFee             = tradingFee,
            lmTable                = lmTable,
            lmTable_stride         = lmTable_stride,
            lmTable_nTiers         = lmTable_nTiers,
            #Base Data
            loop_index              = loop_index,
            data_prices             = data_prices,
            data_prices_stride      = data_prices_stride,
            tp_fsl_immed            = tp_fsl_immed, 
            tp_fsl_close            = tp_fsl_close,
            params_trade_pslReentry = params_trade_pslReentry,
            #State & Result Tensors
            tefDir_this            = tefDir_this,
            tefVal_this            = tefVal_this, 
            balance_cross          = balance_cross,
            balance_isolated       = balance_isolated,
            balance_wallet         = balance_wallet, 
            balance_margin         = balance_margin,
            balance_allocated      = balance_allocated, 
            quantity               = quantity, 
            entryPrice             = entryPrice, 
            forceExited            = forceExited, 
            nTrades                = nTrades, 
            balance_ftIndex        = balance_ftIndex,
            bt_sum                 = bt_sum, 
            bt_sum_xy              = bt_sum_xy,
            bt_sum_squared         = bt_sum_squared, 
            balance_wallet_history = balance_wallet_history, 
            balance_margin_history = balance_margin_history
            )
    #Balance Trend Evaluation
    sf.evaluateBalanceTrend_triton_kernel(
        size_dataLen                 = size_dataLen,
        offsets                      = offsets,
        mask                         = mask,
        balance_wallet               = balance_wallet,
        nTrades                      = nTrades,
        balance_ftIndex              = balance_ftIndex,
        bt_sum                       = bt_sum,
        bt_sum_xy                    = bt_sum_xy,
        bt_sum_squared               = bt_sum_squared,
        balance_finals               = balance_finals,
        balance_bestFit_intercepts   = balance_bestFit_intercepts,
        balance_bestFit_growthRates  = balance_bestFit_growthRates,
        balance_bestFit_volatilities = balance_bestFit_volatilities,
        balance_ftIndexes            = balance_ftIndexes,
        nTrades_rb                   = nTrades_rb
        )
# =======================================================================================================================================================================================================================================================





"""
<Triton Kernel Function>
 * This is an TEF value calculation function written in Triton.
 * It simply takes in model parameters, model state trackers, and base data, and calculate TEF value for trading simulation in the base Triton Kernel Function.
 * This is an example and is recommended to be kept without edits for reference. The user may add similar .py files following the general structure in this file to test their customized strategies. In order for the trade simulator function to be able to 
   recognize and call this function, the user must implement the model parameter import, state trackers initialization, and function call parts for the new specific model. Check 'processBatch_triton_kernel' function in 'exitFunction_base.py'
"""
@triton.jit
def getTEFValue(
    #Process
    size_block: tl.constexpr,
    #Base Data
    loop_index,
    data_analysis, 
    data_analysis_stride: tl.constexpr,
    #Model Parameters
    mp_alpha,
    mp_beta,
    mp_delta,
    mp_strength_S,
    mp_strength_L,
    mp_dhThreshold,
    #Model State Trackers
    st_tefVal_prev,
    st_tefVal_absMax_prev,
    st_mmacd_msDeltaAbsMARel_prev,
    st_isDownhill_prev
    ):

    #[1]: Base Data - Analysis
    analysis_base_ptr_this = data_analysis + (loop_index * data_analysis_stride)
    analysis_mmacd_msDeltaAbsMARel = tl.load(analysis_base_ptr_this + 0)



    #[2]: Nan Check
    isNan = (analysis_mmacd_msDeltaAbsMARel != analysis_mmacd_msDeltaAbsMARel)
    analysis_mmacd_msDeltaAbsMARel = tl.where(isNan, 0.0, analysis_mmacd_msDeltaAbsMARel)



    #[3]: ABSMARel Cycle
    isShort_prev = (st_mmacd_msDeltaAbsMARel_prev  < mp_delta)
    isShort_this = (analysis_mmacd_msDeltaAbsMARel < mp_delta)
    cycleReset   = (isShort_prev ^ isShort_this)



    #[4]: TEF Value Calculation
    #---[4-1]: Effective Params
    mp_strength_eff = tl.where(isShort_this, mp_strength_S, mp_strength_L)

    #---[4-2]: MSDeltaAbsMARel Normalization
    x_sign = tl.where(analysis_mmacd_msDeltaAbsMARel < 0, -1.0, 1.0)
    x_abs  = tl_ld.pow(tl.abs(analysis_mmacd_msDeltaAbsMARel/mp_alpha), mp_beta)
    y_norm = tl_ld.tanh(x_abs)*x_sign

    #---[4-3]: TEF Value
    width = tl.where(isShort_this, mp_delta+1.0, 1.0-mp_delta)
    dist  = tl.abs(y_norm-mp_delta)
    tefVal_this_abs = dist/tl.maximum(width, 1e-9)*mp_strength_eff
    tefVal_this_abs = tl.where(width == 0.0, 0.0, tefVal_this_abs)
    tefVal_this_abs = tl.floor(tefVal_this_abs * 1e6 + 0.5) / 1e6

    #---[4-4]: TEF Value Cyclic Maximum
    st_tefVal_absMax = tl.where(cycleReset, tefVal_this_abs, tl.maximum(st_tefVal_absMax_prev, tefVal_this_abs))

    #---[4-5]: Mode
    st_isDownhill = tl.where(cycleReset, 0.0, st_isDownhill_prev)
    st_isDownhill = tl.where(tefVal_this_abs < (st_tefVal_absMax * mp_dhThreshold), 1.0, st_isDownhill)
    
    #---[4-6]: Effective TEF Value
    tefVal_this_abs = tl.where(st_isDownhill == 1.0, tl.minimum(tefVal_this_abs, tl.abs(st_tefVal_prev)), tefVal_this_abs)

    #---[4-7]: Direction
    tefVal_this = tl.where(isShort_this, -tefVal_this_abs, tefVal_this_abs)

    #---[4-8]: Nan Check
    tefDir_this = tl.where(isNan, 0.0, tl.where(isShort_this, -1.0, 1.0))
    tefVal_this = tl.where(isNan, 0.0, tefVal_this)



    #[5]: State Trackers Update
    st_tefVal_prev                = tl.where(isNan, st_tefVal_prev,                tefVal_this)
    st_tefVal_absMax_prev         = tl.where(isNan, st_tefVal_absMax_prev,         st_tefVal_absMax)
    st_mmacd_msDeltaAbsMARel_prev = tl.where(isNan, st_mmacd_msDeltaAbsMARel_prev, analysis_mmacd_msDeltaAbsMARel)
    st_isDownhill_prev            = tl.where(isNan, st_isDownhill_prev,            st_isDownhill)



    #[6]: Return TEF Value & States
    return (tefDir_this,
            tefVal_this,
            st_tefVal_prev,
            st_tefVal_absMax_prev,
            st_mmacd_msDeltaAbsMARel_prev,
            st_isDownhill_prev)