import triton
import triton.language as tl

TRITON_AUTOTUNE_CONFIGURATIONS = [triton.Config({'size_block': 32},  num_warps= 1, num_stages=2),
                                  triton.Config({'size_block': 32},  num_warps= 1, num_stages=3),
                                  triton.Config({'size_block': 32},  num_warps= 1, num_stages=4),

                                  triton.Config({'size_block': 64},  num_warps= 2, num_stages=2),
                                  triton.Config({'size_block': 64},  num_warps= 2, num_stages=3),
                                  triton.Config({'size_block': 64},  num_warps= 2, num_stages=4),

                                  triton.Config({'size_block': 128}, num_warps= 4, num_stages=2),
                                  triton.Config({'size_block': 128}, num_warps= 4, num_stages=3),
                                  triton.Config({'size_block': 128}, num_warps= 4, num_stages=4),
                                  triton.Config({'size_block': 128}, num_warps= 4, num_stages=5),

                                  triton.Config({'size_block': 256}, num_warps= 8, num_stages=2),
                                  triton.Config({'size_block': 256}, num_warps= 8, num_stages=3),
                                  triton.Config({'size_block': 256}, num_warps= 8, num_stages=4),

                                  triton.Config({'size_block': 512}, num_warps=16, num_stages=2)
                                 ]
TRITON_AUTOTUNE_KEY = ['size_paramsBatch']

PRICEINDEX_HIGHPRICE:  tl.constexpr = 0
PRICEINDEX_LOWPRICE:   tl.constexpr = 1
PRICEINDEX_CLOSEPRICE: tl.constexpr = 2

def processBatch(tkf, **kwargs):
    if kwargs['SEEKERMODE']:
        grid = lambda META: (triton.cdiv(kwargs['size_paramsBatch'], META['size_block']),)
        tkf[grid](**kwargs)
    else:
        kwargs['size_block'] = 32
        kwargs['num_warps']  = 1
        kwargs['num_stages'] = 2
        grid = (triton.cdiv(kwargs['size_paramsBatch'], kwargs['size_block']),)
        tkf.fn[grid](**kwargs)

@triton.jit
def initializeSimulation_triton_kernel(
    balance_initial:        tl.constexpr,
    balance_allocation_max: tl.constexpr,
    params_trade_fslImmed,
    params_trade_fslClose,
    allocationRatio:  tl.constexpr,
    size_paramsBatch: tl.constexpr, 
    size_block:       tl.constexpr
    ):
    #[1]: Offsets
    offsets = tl.program_id(0) * size_block + tl.arange(0, size_block)

    #[2]: Mask
    mask = offsets < size_paramsBatch

    #[3]: Trade Parameters Load
    tp_fsl_immed = tl.load(pointer = params_trade_fslImmed + offsets, mask = mask)
    tp_fsl_close = tl.load(pointer = params_trade_fslClose + offsets, mask = mask)

    #[4]: Trade Simulation State
    balance_wallet    = tl.zeros(shape = [size_block], dtype = tl.float32) + balance_initial
    balance_allocated = tl.minimum(balance_wallet * allocationRatio, balance_allocation_max)
    balance_margin    = tl.zeros(shape = [size_block], dtype = tl.float32) + balance_initial
    balance_ftIndex   = tl.full(shape = [size_block], value = -1, dtype = tl.int32)
    quantity          = tl.zeros(shape = [size_block,], dtype = tl.float32)
    entryPrice        = tl.zeros(shape = [size_block,], dtype = tl.float32)
    forceExited       = tl.zeros(shape = [size_block,], dtype = tl.float32)
    nTrades           = tl.zeros(shape = [size_block,], dtype = tl.float32)

    #[5]: Balance Trend
    bt_sum         = tl.zeros(shape = [size_block,], dtype = tl.float32)
    bt_sum_squared = tl.zeros(shape = [size_block,], dtype = tl.float32)
    bt_sum_xy      = tl.zeros(shape = [size_block,], dtype = tl.float32)

    #[6]: Return Initialized Contents
    return (offsets,
            mask,
            tp_fsl_immed,
            tp_fsl_close,
            balance_wallet,
            balance_allocated,
            balance_margin,
            balance_ftIndex,
            quantity,
            entryPrice,
            forceExited,
            nTrades,
            bt_sum,
            bt_sum_xy,
            bt_sum_squared
            )

@triton.jit
def round_to_step(x, step):
    return tl.where(0 <= x, tl.floor(x/step + 0.5), -tl.floor(-x/step + 0.5)) * step

@triton.jit
def floor_to_step(x, step):
    sign = tl.where(x < 0, -1.0, 1.0)
    return tl.floor(tl.abs(x) / step) * step * sign

@triton.jit
def processTrade_triton_kernel(
    #Process
    offsets, 
    mask, 
    size_dataLen,
    SEEKERMODE: tl.constexpr,
    #Constants
    balance_initial:        tl.constexpr,
    balance_allocation_max: tl.constexpr,
    step_price:             tl.constexpr,
    step_quantity:          tl.constexpr,
    step_quote:             tl.constexpr,
    leverage:               tl.constexpr,
    allocationRatio:        tl.constexpr,
    tradingFee:             tl.constexpr,
    #Base Data
    loop_index,
    data_prices,
    data_prices_stride,
    tp_fsl_immed, 
    tp_fsl_close,
    params_trade_pslReentry: tl.constexpr,
    #State & Result Tensors
    tefDir_this,
    tefVal_this, 
    balance_wallet, 
    balance_allocated, 
    balance_margin,
    quantity, 
    entryPrice, 
    forceExited, 
    nTrades, 
    balance_ftIndex,
    bt_sum, 
    bt_sum_xy,
    bt_sum_squared, 
    balance_wallet_history, 
    balance_margin_history
    ):
    #[1]: Trade Simulation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #---[1-1]: TEF Range Control
    tefVal_this = tl.minimum(tefVal_this,  1.0)
    tefVal_this = tl.maximum(tefVal_this, -1.0)

    #---[1-2]: Prices
    price_base_ptr_this = data_prices + (loop_index * data_prices_stride)
    price_high  = tl.load(price_base_ptr_this + PRICEINDEX_HIGHPRICE)
    price_low   = tl.load(price_base_ptr_this + PRICEINDEX_LOWPRICE)
    price_close = tl.load(price_base_ptr_this + PRICEINDEX_CLOSEPRICE)

    #---[1-3]: Position Side & Has #qty_entry
    position_side = tl.where(0 < quantity,  1.0, 0.0)
    position_side = tl.where(quantity < 0, -1.0, position_side)
    position_has = (quantity != 0)

    #---[1-4]: Exit Conditions
    price_act_FSLImmed = round_to_step(entryPrice * (1.0 - position_side*tp_fsl_immed), step_price)
    price_act_FSLClose = round_to_step(entryPrice * (1.0 - position_side*tp_fsl_close), step_price)
    price_liquidation  = round_to_step(entryPrice * (1.0 - position_side/leverage),     step_price)
    price_worst = tl.where(0 < quantity, price_low,  price_close)
    price_worst = tl.where(quantity < 0, price_high, price_worst)
    hit_liquidation  = position_has & ((position_side*price_worst) <= (position_side*price_liquidation))
    hit_fslImmed     = position_has & ((position_side*price_worst) <= (position_side*price_act_FSLImmed))
    hit_fslClose     = position_has & ((position_side*price_close) <= (position_side*price_act_FSLClose))
    status_forceExit = hit_liquidation | hit_fslImmed | hit_fslClose
    status_clear     = (position_side != tefDir_this)

    #---[1-5]: Exit Execution Price
    dist_fslImmed        = tl.abs(entryPrice - price_act_FSLImmed)
    dist_liq             = tl.abs(entryPrice - price_liquidation)
    price_intra_first    = tl.where(dist_fslImmed < dist_liq, price_act_FSLImmed, price_liquidation)
    hit_any_intra        = hit_liquidation | hit_fslImmed
    price_exit_execution = price_close
    price_exit_execution = tl.where(hit_fslClose, price_act_FSLClose, price_exit_execution)
    price_exit_execution = tl.where(hit_any_intra, price_intra_first, price_exit_execution)
    price_exit_execution = round_to_step(price_exit_execution, step_price)
    
    #---[1-6]: Quantity Reduce
    balance_toCommit  = balance_allocated * tl.abs(tefVal_this)
    balance_committed = tl.abs(quantity) * entryPrice / leverage
    balance_toExit    = tl.maximum(balance_committed-balance_toCommit, 0.0)
    quantity_reduce   = tl.where(status_forceExit | status_clear,
                                 tl.abs(quantity),
                                 floor_to_step(balance_toExit * leverage / tl.maximum(entryPrice, 1e-12), step_quantity))
    profit       = round_to_step(quantity_reduce * (price_exit_execution - entryPrice) * position_side, step_quote)
    fee          = round_to_step(quantity_reduce * price_exit_execution * tradingFee,                   step_quote)
    quantity_new = round_to_step(quantity-quantity_reduce*position_side, step_quantity)

    #---[1-7]: Post-Exit Wallet & Allocated Balance Update
    balance_wallet = balance_wallet + (profit - fee)
    balance_wallet = tl.maximum(balance_wallet, 0.0)
    balance_wallet = round_to_step(balance_wallet, step_quote)
    balance_allocated = tl.where(quantity_new == 0.0, 
                                 tl.minimum(balance_wallet*allocationRatio, balance_allocation_max), 
                                 balance_allocated)
    
    #---[1-8]: Force Exit State Update
    if not params_trade_pslReentry:
        forceExited = tl.where(status_forceExit,           position_side, forceExited)
        forceExited = tl.where(forceExited != tefDir_this, 0.0,           forceExited)

    #---[1-9]: Quantity Increase
    balance_toCommit  = balance_allocated * tl.abs(tefVal_this)
    balance_committed = tl.abs(quantity_new) * entryPrice / leverage
    balance_toEnter   = tl.maximum(balance_toCommit-balance_committed, 0.0)
    quantity_entry = tl.where(forceExited == 0.0,
                              floor_to_step(balance_toEnter * leverage / price_close, step_quantity),
                              0.0)
    fee            = round_to_step(tl.abs(quantity_entry) * price_close * tradingFee, step_quote)
    quantity_final = round_to_step(quantity_new + quantity_entry*tefDir_this, step_quantity)
    
    #---[1-10]: Entry Price Update
    entryPrice_new = tl.where(quantity_final == 0.0, 
                              0.0, 
                              (tl.abs(quantity_new)*entryPrice + quantity_entry*price_close) / tl.maximum(tl.abs(quantity_final), 1e-6))
    entryPrice_new = round_to_step(entryPrice_new, step_price)
    
    #---[1-11]: Post-Entry Wallet Balance Update
    balance_wallet = round_to_step(balance_wallet - fee, 
                                   step_quote)
    balance_wallet = tl.maximum(balance_wallet, 0.0)

    #---[1-12]: Margin Balance
    balance_margin = round_to_step(balance_wallet + quantity_final * (price_close - entryPrice_new), 
                                   step_quote)
    balance_margin = tl.maximum(balance_margin, 0.0)

    #---[1-13]: Update State
    balance_ftIndex = tl.where((balance_ftIndex == -1) & (quantity_final != quantity), loop_index, balance_ftIndex)
    trade_exit  = tl.where(0.0 < quantity_reduce, 1, 0)
    trade_entry = tl.where(0.0 < quantity_entry,  1, 0)
    nTrades     = nTrades + trade_exit + trade_entry
    quantity   = quantity_final
    entryPrice = entryPrice_new
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #[2]: Balance Trend Trackers ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    first_trade_occurred = (0 <= balance_ftIndex)
    bt_val_x = tl.where(first_trade_occurred, (loop_index-balance_ftIndex).to(tl.float32), 0.0)
    bt_val_y = tl.where(first_trade_occurred, 
                        tl.log(tl.maximum(balance_wallet, 1e-9) / balance_initial),
                        0.0)
    bt_sum         += bt_val_y
    bt_sum_xy      += bt_val_x*bt_val_y
    bt_sum_squared += bt_val_y*bt_val_y
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #[3]: History Record ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not SEEKERMODE:
        off_write = offsets * size_dataLen + loop_index
        tl.store(pointer = balance_wallet_history  + off_write, value = balance_wallet, mask = mask)
        tl.store(pointer = balance_margin_history  + off_write, value = balance_margin, mask = mask)
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    return (
        balance_wallet, 
        balance_allocated, 
        balance_margin,
        balance_ftIndex,
        quantity, 
        entryPrice, 
        forceExited, 
        nTrades,
        bt_sum, 
        bt_sum_xy,
        bt_sum_squared
    )

@triton.jit
def evaluateBalanceTrend_triton_kernel(
    size_dataLen: tl.constexpr,
    offsets,
    mask,
    balance_wallet,
    nTrades,
    balance_ftIndex,
    bt_sum,
    bt_sum_xy,
    bt_sum_squared,
    balance_finals,
    balance_bestFit_intercepts,
    balance_bestFit_growthRates,
    balance_bestFit_volatilities,
    balance_ftIndexes,
    nTrades_rb
    ):
    bt_n      = (size_dataLen-balance_ftIndex).to(tl.float32)
    bt_valid  = (0 <= balance_ftIndex) & (1.0 < bt_n)
    bt_n_safe = tl.where(bt_valid, bt_n, 1.0)

    bt_sum_x  = bt_n*(bt_n-1.0)*0.5
    bt_mean_y = bt_sum   / bt_n_safe
    bt_mean_x = bt_sum_x / bt_n_safe

    denominator_growth      = (bt_n * bt_n * (bt_n * bt_n - 1.0)) / 12.0
    denominator_growth_safe = tl.where(bt_valid, denominator_growth, 1.0)
    numerator_growth        = (bt_n * bt_sum_xy) - (bt_sum_x * bt_sum)
    raw_growthRate          = numerator_growth / denominator_growth_safe

    raw_intercepts = bt_mean_y - (raw_growthRate * bt_mean_x)

    bt_var_x = (bt_n * bt_n - 1.0) / 12.0
    bt_var_y = (bt_sum_squared / bt_n_safe) - (bt_mean_y * bt_mean_y)
    
    raw_variance_resid = tl.maximum(bt_var_y - (raw_growthRate * raw_growthRate * bt_var_x), 0.0)
    raw_volatility     = tl.sqrt(raw_variance_resid)

    bt_growthRate = tl.where(bt_valid, raw_growthRate, 0.0)
    bt_intercepts = tl.where(bt_valid, raw_intercepts, 0.0)
    bt_volatility = tl.where(bt_valid, raw_volatility, 0.0)

    #Final Results Store
    tl.store(pointer = balance_bestFit_intercepts   + offsets, value = bt_intercepts,   mask = mask)
    tl.store(pointer = balance_bestFit_growthRates  + offsets, value = bt_growthRate,   mask = mask)
    tl.store(pointer = balance_bestFit_volatilities + offsets, value = bt_volatility,   mask = mask)
    tl.store(pointer = balance_finals               + offsets, value = balance_wallet,  mask = mask)
    tl.store(pointer = balance_ftIndexes            + offsets, value = balance_ftIndex, mask = mask)
    tl.store(pointer = nTrades_rb                   + offsets, value = nTrades,         mask = mask)

