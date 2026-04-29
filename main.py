#Python Modules
import time
import numpy
import datetime
import json
import os
import matplotlib
import matplotlib.pyplot
import termcolor
import math
import config
from rich.console import Console as rConsole
from rich.live    import Live    as rLive

#TEFFP Seeker Modules
from exitFunction_base import exitFunction

#Leverage Margin Table
try:
    with open(file     = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'binance_futures_leverage_margin_table.json'), 
              mode     = 'r',
              encoding = 'utf-8') as f:
        LEVERAGEMARGINTABLE = json.load(f)['table']
except Exception as e:
    print(termcolor.colored("[WARNING - LEVERAGE MARGIN TABLE NOT FOUND]", 'light_red'))
    LEVERAGEMARGINTABLE = dict()





#TEST FUNCTION ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test(config_test):
    #[1]: System Message
    print(termcolor.colored("[PARAMETER TEST]", 'light_blue'))

    #[2]: Directories
    dir_file_base       = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')
    dir_file_descriptor = os.path.join(dir_file_base, config_test['analysisData']+'_descriptor.json')
    dir_file_data       = os.path.join(dir_file_base, config_test['analysisData']+'_data.npy')

    #[3]: Data Load
    with open(dir_file_descriptor, 'r', encoding = 'utf-8') as _f: 
        descriptor = json.loads(_f.read())
    linearizedAnalysis = numpy.load(dir_file_data)

    #[4]: eFunction Initialization
    eFunction = exitFunction(modelName              = config_test['exitFunctionType'],
                             isSeeker               = False, 
                             balance_initial        = config_test['balance_initial'],
                             balance_allocation_max = config_test['balance_allocation_max'],
                             leverage               = config_test['leverage'], 
                             isolated               = config_test['isolated'],
                             pslReentry             = config_test['pslReentry'],
                             precision_price        = descriptor['pricePrecision'],
                             precision_quantity     = descriptor['quantityPrecision'],
                             precision_quote        = descriptor['quotePrecision'],
                             lmTable                = LEVERAGEMARGINTABLE.get(descriptor['positionSymbol'], None))
    ppResult = eFunction.preprocessData(linearizedAnalysis = linearizedAnalysis, indexIdentifier = descriptor['indexIdentifier'])
    if ppResult is None:
        print(termcolor.colored("[PARAMETER TEST FAILED]", 'light_red'))
        return
    validityRate, gapRate = ppResult
    print(f" - PREPROCESS RESULT")
    print(f"   - Validity Rate: {validityRate*100:.3f} %")
    print(f"   - Gap Rate:      {gapRate*100:.3f} %")

    #[5]: eFunction Processing
    (
        balance_wallet_history, 
        balance_margin_history, 
        balance_bestFit_history,
        balance_bestFit_growthRates,
        balance_bestFit_volatilities,
        nTrades
    ) = eFunction.performOnParams(params = [config_test['tradeParams']+config_test['modelParams'],])

    #[6]: Matplot Drawing
    matplotlib.pyplot.plot(balance_wallet_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 1.0), linestyle='solid',  linewidth=1)
    matplotlib.pyplot.plot(balance_margin_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 0.5), linestyle='dashed', linewidth=1)
    matplotlib.pyplot.plot(balance_bestFit_history[0,:].cpu(), color=(0.8, 0.5, 0.8, 1.0), linestyle='solid',  linewidth=1)

    #[7]: Summary
    balance_final        = balance_wallet_history[0,-1].item()
    balance_final_growth = balance_final/config_test['balance_initial']-1
    growthRate_interval  = balance_bestFit_growthRates[0].item()
    growthRate_daily     = math.exp(growthRate_interval*1440)        -1
    growthRate_monthly   = math.exp(growthRate_interval*1440*30.4167)-1
    volatility           = balance_bestFit_volatilities[0].item()
    volatility_tMin_997  = math.exp(-volatility*3)-1
    volatility_tMax_997  = math.exp( volatility*3)-1
    print(f" - TEST RESULT")
    if   balance_final_growth < 0:  print(f"   - Final Balance: {balance_final:.8f}", termcolor.colored(f"[{balance_final_growth*100:.3f} %]", 'light_red'))
    elif balance_final_growth == 0: print(f"   - Final Balance: {balance_final:.8f}", termcolor.colored(f"[{balance_final_growth*100:.3f} %]", None))
    else:                           print(f"   - Final Balance: {balance_final:.8f}", termcolor.colored(f"[+{balance_final_growth*100:.3f} %]", 'light_green'))
    if   growthRate_daily < 0:  print(f"   - Growth Rate:   {growthRate_interval:.8f} / {termcolor.colored(f'{growthRate_daily*100:.3f} %', 'light_red')} [Daily] / {termcolor.colored(f'{growthRate_monthly*100:.3f} %', 'light_red')} [Monthly]")
    elif growthRate_daily == 0: print(f"   - Growth Rate:   {growthRate_interval:.8f} / {termcolor.colored(f'{growthRate_daily*100:.3f} %', None)} [Daily] / {termcolor.colored(f'{growthRate_monthly*100:.3f} %', None)} [Monthly]")
    else:                       print(f"   - Growth Rate:   {growthRate_interval:.8f} / {termcolor.colored(f'+{growthRate_daily*100:.3f} %', 'light_green')} [Daily] / {termcolor.colored(f'+{growthRate_monthly*100:.3f} %', 'light_green')} [Monthly]")
    print(f"   - Volatility:    {volatility:.8f} [Theoretical 99.7%: {termcolor.colored(f'{volatility_tMin_997*100:.3f} %', 'light_magenta')} / {termcolor.colored(f'+{volatility_tMax_997*100:.3f} %', 'light_blue')}]")
    print(f"   - nTrades:       {int(nTrades[0].item())}")

    #[8]: Matplot Show
    matplotlib.pyplot.title(f"[PARAMETER TEST] Wallet & Margin Balance History")
    matplotlib.pyplot.show()

    #[9]: System Message
    print(termcolor.colored("[PARAMETER TEST COMPLETE]", 'light_blue'))
#TEST FUNCTION END ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#SEEK FUNCTION ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def seek(config_seek, process_begin_time):
    #[1]: System Message
    print(termcolor.colored("[Seeker]", 'light_blue'))

    #[2]: Process Preparation
    #---[2-1]: Analysis Code
    rCode = f"teffps_result_{int(process_begin_time)}"
    print(f" * Result Code: {rCode}")

    #---[2-2]: Results Buffer
    processes = dict()
    for pIndex, st in enumerate(config_seek):
        #[2-2-1]: Files Existence Check
        dir_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')
        dir_file_descriptor = os.path.join(dir_file_base, st['analysisData']+'_descriptor.json')
        dir_file_data       = os.path.join(dir_file_base, st['analysisData']+'_data.npy')
        if not os.path.isfile(dir_file_descriptor) or not os.path.isfile(dir_file_data): continue

        #[2-2-2]: Descriptor Read
        dir_file_descriptor = os.path.join(dir_file_base, st['analysisData']+'_descriptor.json')
        dir_file_data       = os.path.join(dir_file_base, st['analysisData']+'_data.npy')
        with open(dir_file_descriptor, 'r', encoding = 'utf-8') as f: 
            descriptor = json.loads(f.read())

        #[2-2-3]: Processes
        processes[pIndex] = {'descriptor': descriptor.copy(),
                             'bestResult': None,
                             'records':    list()}
            
    #[3]: Results Generation
    print(" * Seeker Process")
    for pIndex, process in processes.items():
        #[3-1]: Configuration
        st = config_seek[pIndex]

        #[3-2]: System Message
        print(f"  [{pIndex+1} / {len(processes)}] <ANALYSIS DATA - '{st['analysisData']}'>")

        #[3-3]: Directories
        dir_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')
        dir_file_descriptor = os.path.join(dir_file_base, st['analysisData']+'_descriptor.json')
        dir_file_data       = os.path.join(dir_file_base, st['analysisData']+'_data.npy')

        #[3-4]: Data Load
        with open(dir_file_descriptor, 'r', encoding = 'utf-8') as f: 
            descriptor = json.loads(f.read())
        linearizedAnalysis = numpy.load(dir_file_data)

        #[3-5]: eFunction Initialization
        eFunction = exitFunction(modelName              = st['exitFunctionType'],
                                 isSeeker               = True, 
                                 balance_initial        = st['balance_initial'],
                                 balance_allocation_max = st['balance_allocation_max'],
                                 leverage               = st['leverage'],
                                 isolated               = st['isolated'],
                                 pslReentry             = st['pslReentry'],
                                 precision_price        = descriptor['pricePrecision'],
                                 precision_quantity     = descriptor['quantityPrecision'],
                                 precision_quote        = descriptor['quotePrecision'],
                                 lmTable                = LEVERAGEMARGINTABLE.get(descriptor['positionSymbol'], None))
        print(f"    - Preprocessing Analysis Data...")
        t_0 = time.perf_counter_ns()
        ppResult = eFunction.preprocessData(linearizedAnalysis = linearizedAnalysis, indexIdentifier = descriptor['indexIdentifier'])
        if ppResult is None:
            print(termcolor.colored(f"    - Analysis Data Preprocessing Failed. This Process {pIndex} Will Be Skipped.", 'light_magenta'))
            continue
        t_1 = time.perf_counter_ns()
        validityRate, gapRate = ppResult
        print( "    - Analysis Data Preprocessing Complete! ")
        print(f"      - Elapsed Time:  {(t_1-t_0)/1e6:.3f} ms")
        print(f"      - Validity Rate: {validityRate*100:.3f} %")
        print(f"      - Gap Rate:      {gapRate*100:.3f} %")
        asp = eFunction.initializeSeeker(tradeParamConfig         = st['tradeParamConfig'], 
                                         modelParamConfig         = st['modelParamConfig'], 
                                         nSeekerPoints            = st['nSeekerPoints'],
                                         parameterBatchSize       = st['parameterBatchSize'], 
                                         nRepetition              = st['nRepetition'],
                                         learningRate             = st['learningRate'],
                                         deltaRatio               = st['deltaRatio'],
                                         beta_velocity            = st['beta_velocity'],
                                         beta_momentum            = st['beta_momentum'],
                                         repopulationRatio        = st['repopulationRatio'],
                                         repopulationInterval     = st['repopulationInterval'],
                                         repopulationGuideRatio   = st['repopulationGuideRatio'],
                                         repopulationDecayRate    = st['repopulationDecayRate'],
                                         scoring                  = st['scoring'], 
                                         scoring_maxMDD           = st['scoring_maxMDD'],
                                         scoring_growthRateWeight = st['scoring_growthRateWeight'],
                                         scoring_growthRateScaler = st['scoring_growthRateScaler'],
                                         scoring_volatilityWeight = st['scoring_volatilityWeight'],
                                         scoring_volatilityScaler = st['scoring_volatilityScaler'],
                                         scoring_nTradesWeight    = st['scoring_nTradesWeight'],
                                         scoring_nTradesScaler    = st['scoring_nTradesScaler'],
                                         scoringSamples           = st['scoringSamples'], 
                                         terminationThreshold     = st['terminationThreshold'], 
                                        )
        print(f"    - eFunction Initialization Complete!")
        print(f"      - Exit Function Type:            {st['exitFunctionType']}")
        print(f"      - Initial Balance:               {st['balance_initial']}")
        print(f"      - Maximum Balance Allocation:    {st['balance_allocation_max']}")
        print(f"      - Leverage:                      {st['leverage']}")
        print(f"      - Isolated:                      {st['isolated']}")
        print(f"      - PSL Re-entry:                  {st['pslReentry']}")
        try:    st['tradeParamConfig'] = tuple(st['tradeParamConfig'])
        except: pass
        if asp['tradeParamConfig']         != st['tradeParamConfig']:         print(f"      - Trade Parameter Configuration: {st['tradeParamConfig']} -> {asp['tradeParamConfig']}")
        else:                                                                 print(f"      - Trade Parameter Configuration: {asp['tradeParamConfig']}")
        try:    st['modelParamConfig'] = tuple(st['modelParamConfig'])
        except: pass
        if asp['modelParamConfig']         != st['modelParamConfig']:         print(f"      - Model Parameter Configuratio:  {st['modelParamConfig']} -> {asp['modelParamConfig']}")
        else:                                                                 print(f"      - Model Parameter Configuratio:  {asp['modelParamConfig']}")
        if asp['nSeekerPoints']            != st['nSeekerPoints']:            print(f"      - Number of Seeker Points:       {st['nSeekerPoints']} -> {asp['nSeekerPoints']}")
        else:                                                                 print(f"      - Number of Seeker Points:       {asp['nSeekerPoints']}")
        if asp['parameterBatchSize']       != st['parameterBatchSize']:       print(f"      - Parameter Batch Size:          {st['parameterBatchSize']} -> {asp['parameterBatchSize']}")
        else:                                                                 print(f"      - Parameter Batch Size:          {asp['parameterBatchSize']}")
        if asp['nRepetition']              != st['nRepetition']:              print(f"      - Number of Repetition:          {st['nRepetition']} -> {asp['nRepetition']}")
        else:                                                                 print(f"      - Number of Repetition:          {asp['nRepetition']}")
        if asp['learningRate']             != st['learningRate']:             print(f"      - Learning Rate:                 {st['learningRate']} -> {asp['learningRate']}")
        else:                                                                 print(f"      - Learning Rate:                 {asp['learningRate']}")
        if asp['deltaRatio']               != st['deltaRatio']:               print(f"      - Delta Ratio:                   {st['deltaRatio']} -> {asp['deltaRatio']}")
        else:                                                                 print(f"      - Delta Ratio:                   {asp['deltaRatio']}")
        if asp['beta_velocity']            != st['beta_velocity']:            print(f"      - Velocity Beta:                 {st['beta_velocity']} -> {asp['beta_velocity']}")
        else:                                                                 print(f"      - Velocity Beta:                 {asp['beta_velocity']}")
        if asp['beta_momentum']            != st['beta_momentum']:            print(f"      - Momentum Beta:                 {st['beta_momentum']} -> {asp['beta_momentum']}")
        else:                                                                 print(f"      - Momentum Beta:                 {asp['beta_momentum']}")
        if asp['repopulationRatio']        != st['repopulationRatio']:        print(f"      - Repopulation Ratio:            {st['repopulationRatio']} -> {asp['repopulationRatio']}")
        else:                                                                 print(f"      - Repopulation Ratio:            {asp['repopulationRatio']}")
        if asp['repopulationInterval']     != st['repopulationInterval']:     print(f"      - Repopulation Interval:         {st['repopulationInterval']} -> {asp['repopulationInterval']}")
        else:                                                                 print(f"      - Repopulation Interval:         {asp['repopulationInterval']}")
        if asp['repopulationGuideRatio']   != st['repopulationGuideRatio']:   print(f"      - Repopulation Guide Ratio:      {st['repopulationGuideRatio']} -> {asp['repopulationGuideRatio']}")
        else:                                                                 print(f"      - Repopulation Guide Ratio:      {asp['repopulationGuideRatio']}")
        if asp['repopulationDecayRate']    != st['repopulationDecayRate']:    print(f"      - Repopulation Decay Rate:       {st['repopulationDecayRate']} -> {asp['repopulationDecayRate']}")
        else:                                                                 print(f"      - Repopulation Decay Rate:       {asp['repopulationDecayRate']}")
        if asp['scoring']                  != st['scoring']:                  print(f"      - Scoring:                       {st['scoring']} -> {asp['scoring']}")
        else:                                                                 print(f"      - Scoring:                       {asp['scoring']}")
        if asp['scoring_maxMDD']           != st['scoring_maxMDD']:           print(f"      - Scoring Maximum MDD:           {st['scoring_maxMDD']} -> {asp['scoring_maxMDD']}")
        else:                                                                 print(f"      - Scoring Maximum MDD:           {asp['scoring_maxMDD']}")
        if asp['scoring_growthRateWeight'] != st['scoring_growthRateWeight']: print(f"      - Scoring Growth Rate Weight:    {st['scoring_growthRateWeight']} -> {asp['scoring_growthRateWeight']}")
        else:                                                                 print(f"      - Scoring Growth Rate Weight:    {asp['scoring_growthRateWeight']}")
        if asp['scoring_growthRateScaler'] != st['scoring_growthRateScaler']: print(f"      - Scoring Growth Rate Scaler:    {st['scoring_growthRateScaler']} -> {asp['scoring_growthRateScaler']}")
        else:                                                                 print(f"      - Scoring Growth Rate Scaler:    {asp['scoring_growthRateScaler']}")
        if asp['scoring_volatilityWeight'] != st['scoring_volatilityWeight']: print(f"      - Scoring Volatility Weight:     {st['scoring_volatilityWeight']} -> {asp['scoring_volatilityWeight']}")
        else:                                                                 print(f"      - Scoring Volatility Weight:     {asp['scoring_volatilityWeight']}")
        if asp['scoring_volatilityScaler'] != st['scoring_volatilityScaler']: print(f"      - Scoring Volatility Scaler:     {st['scoring_volatilityScaler']} -> {asp['scoring_volatilityScaler']}")
        else:                                                                 print(f"      - Scoring Volatility Scaler:     {asp['scoring_volatilityScaler']}")
        if asp['scoring_nTradesWeight']    != st['scoring_nTradesWeight']:    print(f"      - Scoring nTrades Weight:        {st['scoring_nTradesWeight']} -> {asp['scoring_nTradesWeight']}")
        else:                                                                 print(f"      - Scoring nTrades Weight:        {asp['scoring_nTradesWeight']}")
        if asp['scoring_nTradesScaler']    != st['scoring_nTradesScaler']:    print(f"      - Scoring nTrades Scaler:        {st['scoring_nTradesScaler']} -> {asp['scoring_nTradesScaler']}")
        else:                                                                 print(f"      - Scoring nTrades Scaler:        {asp['scoring_nTradesScaler']}")
        if asp['terminationThreshold']     != st['terminationThreshold']:     print(f"      - Termination Threshold:         {st['terminationThreshold']} -> {asp['terminationThreshold']}")
        else:                                                                 print(f"      - Termination Threshold:         {asp['terminationThreshold']}")
        st['nSeekerPoints']            = asp['nSeekerPoints']
        st['nRepetition']              = asp['nRepetition']
        st['learningRate']             = asp['learningRate']
        st['deltaRatio']               = asp['deltaRatio']
        st['beta_velocity']            = asp['beta_velocity']
        st['beta_momentum']            = asp['beta_momentum']
        st['repopulationRatio']        = asp['repopulationRatio']
        st['repopulationInterval']     = asp['repopulationInterval']
        st['scoring']                  = asp['scoring']
        st['scoring_maxMDD']           = asp['scoring_maxMDD']
        st['scoring_growthRateWeight'] = asp['scoring_growthRateWeight']
        st['scoring_growthRateScaler'] = asp['scoring_growthRateScaler']
        st['scoring_volatilityWeight'] = asp['scoring_volatilityWeight']
        st['scoring_volatilityScaler'] = asp['scoring_volatilityScaler']
        st['scoring_nTradesWeight']    = asp['scoring_nTradesWeight']
        st['scoring_nTradesScaler']    = asp['scoring_nTradesScaler']
        st['scoringSamples']           = asp['scoringSamples']
        st['terminationThreshold']     = asp['terminationThreshold']

        #[3-6]: AutoTune Warm Up
        print(f"    - Warming Up eFunction Autotune (This Could Take A Few Minutes)...")
        atwu_result = eFunction.warmupAutotune()
        if atwu_result[0]: 
            print(f"      - eFunction Autotune Warmup Complete! {atwu_result[1]}")
        else:              
            print(termcolor.colored(f"      - eFunction Autotune Warmup Failed - {atwu_result[1]}", 'light_red'))
            continue

        #[3-7]: Seeker
        print(f"    - Seeking eFunction Optimized Parameters...")
        console    = rConsole(highlight=False)
        complete   = False
        bestResult = None
        try:
            with rLive(console = console, refresh_per_second = 4) as rl:
                while not complete:
                    #Processing
                    (complete, 
                     repetitionIndex, 
                     step, 
                     _bestResult, 
                     t_processing_paramsSet_sim_ms,
                     t_processing_paramsSet_ppp_ms
                    ) = eFunction.runSeeker()
                    _bestResult = {'repetitionIndex': repetitionIndex,
                                   'tradeParams':     _bestResult[0], 
                                   'modelParams':     _bestResult[1], 
                                   'finalBalance':    _bestResult[2],
                                   'growthRate':      _bestResult[3],
                                   'volatility':      _bestResult[4],
                                   'score':           _bestResult[5],
                                   'nTrades':         _bestResult[6]}
                    #Best Result Check
                    if (bestResult is None) or (bestResult['score'] < _bestResult['score']): 
                        bestResult = _bestResult
                        processes[pIndex]['bestResult'] = bestResult
                        processes[pIndex]['records'].append(_bestResult)
                    #Progress Print
                    balance_final        = bestResult['finalBalance']
                    balance_final_growth = balance_final/st['balance_initial']-1
                    if balance_final_growth < 0:
                        bfgStr_color = "bright_red"
                        bfgStr_sign  = ""
                    elif balance_final_growth == 0:
                        bfgStr_color = "white"
                        bfgStr_sign  = ""
                    else:
                        bfgStr_color = "bright_green"
                        bfgStr_sign  = "+"
                    growthRate_interval = bestResult['growthRate']
                    growthRate_daily    = math.exp(growthRate_interval*1440)        -1
                    growthRate_monthly  = math.exp(growthRate_interval*1440*30.4167)-1
                    if   (growthRate_daily < 0):  
                        grStr_color = "bright_red"
                        grStr_sign  = ""
                    elif (growthRate_daily == 0): 
                        grStr_color = "white"
                        grStr_sign  = ""
                    else:                          
                        grStr_color = "bright_green"
                        grStr_sign  = "+"
                    volatility = bestResult['volatility']
                    volatility_tMin_997 = math.exp(-volatility*3)-1
                    volatility_tMax_997 = math.exp( volatility*3)-1
                    ppstr = (f"      - Progress:                       <Repetition: {repetitionIndex+1}/{st['nRepetition']}> <Step: {step}>\n"
                             f"      - Parameter Set Processing Speed: {t_processing_paramsSet_sim_ms*1e3:.3f} us [SIMULATION], {t_processing_paramsSet_ppp_ms*1e3:.3f} us [PRE/POST PROCESSING]\n"
                             f"      - Trade Parameters:               {bestResult['tradeParams']}\n"
                             f"      - Model Parameters:               {bestResult['modelParams']}\n"
                             f"      - Final Balance:                  {balance_final:.8f} [{bfgStr_color}][{bfgStr_sign}{balance_final_growth*100:.3f} %][/]\n"
                             f"      - Growth Rate:                    [{grStr_color}]{grStr_sign}{growthRate_interval:.8f} [/]/ [{grStr_color}]{grStr_sign}{growthRate_daily*100:.3f} % [/][Daily] / [{grStr_color}]{grStr_sign}{growthRate_monthly*100:.3f} % [/][Monthly]\n"
                             f"      - Volatility:                     {volatility:.8f} [Theoretical 99.7%: [bright_magenta]{volatility_tMin_997*100:.3f} % [/]/ [bright_cyan]{volatility_tMax_997*100:.3f} %][/]\n"
                             f"      - Score:                          {bestResult['score']:.8f}\n"
                             f"      - nTrades:                        {bestResult['nTrades']}"
                            )
                    rl.update(ppstr)
        except KeyboardInterrupt:
            print("    - Keyboard Interruption Detected, Terminating...")

        #[3-8]: System Message
        print(f"    - eFunction Optimized Parameters Seeking Process Complete!")

    #[4]: Results Save
    #---[4-1]: Folder Generation
    thisResult_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', rCode)
    os.mkdir(thisResult_path)

    #---[4-2]: Results Save
    pResults = dict()
    for pIndex, process in processes.items():
        if process['bestResult'] is None:
            continue
        pResults[pIndex] = {'config':         config_seek[pIndex],
                            'genTime_ns':     process['descriptor']['genTime_ns'],
                            'simulationCode': process['descriptor']['simulationCode'],
                            'positionSymbol': process['descriptor']['positionSymbol'],
                            'bestResult':     process['bestResult'],
                            'records':        process['records']}
    resultData = {'rCode':   rCode,
                  'time':    datetime.datetime.fromtimestamp(process_begin_time).strftime("%Y/%m/%d %H:%M:%S"),
                  'results': pResults}
    resultData_path = os.path.join(thisResult_path, f"{rCode}_result.json")
    with open(resultData_path, "w", encoding = 'utf-8') as f: 
        json.dump(resultData, f, indent=4)
    print(" * Seeker Result Saved")

    #[5]: Trade Congiruation Export
    for pIndex, process in processes.items():
        bResult = process['bestResult']
        if bResult is None:
            continue
        st = config_seek[pIndex]
        tc = {"leverage":              st['leverage'],
              "isolated":              st['isolated'],
              "direction":             "BOTH",
              "fullStopLossImmediate": bResult['tradeParams'][0],
              "fullStopLossClose":     bResult['tradeParams'][1],
              "postStopLossReentry":   st['pslReentry'],
              "teff_functionType":     st['exitFunctionType'],
              "teff_functionParams":   list(bResult['modelParams'])
             }
        tc_path = os.path.join(thisResult_path, f"{rCode}_{pIndex}_tc.tc")
        with open(tc_path, "w", encoding = 'utf-8') as f:
            json.dump(tc, f, indent=4)
    print(" * Best Result Trade Configurations Saved")

    #[6]: System Message
    print(termcolor.colored("[Seeker Complete]\n", 'light_blue'))

    #[7]: Result Return
    return rCode
#SEEK FUNCTION END ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#READ FUNCTION ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def read(rCord_read):
    #[1]: System Message
    print(termcolor.colored("[Seeker Result Read]", 'light_blue'))

    #[2]: Directories
    dir_result_folder  = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', rCord_read)
    dir_result_file    = os.path.join(dir_result_folder, f"{rCord_read}_result.json")
    dir_data_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')

    #[3]: Result Data
    with open(dir_result_file, 'r', encoding = 'utf-8') as _f: 
        resultData = json.loads(_f.read())
    print(f" * Result Code: {resultData['rCode']}")
    print(f" * Time:        {resultData['time']}")
    print(f" * Results:")

    #[4]: Result Data Display
    for pIndexstr, pResult in resultData['results'].items():
        #[4-1]: Configuration 
        st = pResult['config']

        #[4-2]: System Message
        print(f"  [{int(pIndexstr)+1} / {len(resultData['results'])}] <ANALYSIS DATA - '{st['analysisData']}'>")

        #[4-3]: Analysis Data Directories
        dir_file_descriptor = os.path.join(dir_data_file_base, st['analysisData']+'_descriptor.json')
        dir_file_data       = os.path.join(dir_data_file_base, st['analysisData']+'_data.npy')

        #[4-4]: Analysis Data Load
        with open(dir_file_descriptor, 'r', encoding = 'utf-8') as _f: 
            descriptor = json.loads(_f.read())
        linearizedAnalysis = numpy.load(dir_file_data)

        #[4-5]: Analysis Data Match Check
        if ((pResult['genTime_ns']     == descriptor['genTime_ns'])     & \
            (pResult['simulationCode'] == descriptor['simulationCode']) & \
            (pResult['positionSymbol'] == descriptor['positionSymbol'])):
            print("    - Analysis Data Match:          ", termcolor.colored("TRUE", 'light_green'))
        else:
            print("    - Analysis Data Match:          ", termcolor.colored("FALSE", 'light_red'))

        #[4-6]: Process Parameters
        print(f"    - Exit Function Type:            {st['exitFunctionType']}")
        print(f"    - Initial Balance:               {st['balance_initial']}")
        print(f"    - Maximum Balance Allocation:    {st['balance_allocation_max']}")
        print(f"    - Leverage:                      {st['leverage']}")
        print(f"    - Isolated:                      {st['isolated']}")
        print(f"    - PSL Re-Entry:                  {st['pslReentry']}")
        print(f"    - Trade Parameter Configuration: {tuple(st['tradeParamConfig'])}")
        print(f"    - Model Parameter Configuration: {tuple(st['modelParamConfig'])}")
        print(f"    - Number of Seeker Points:       {st['nSeekerPoints']}")
        print(f"    - Parameter Batch Size:          {st['parameterBatchSize']}")
        print(f"    - Number of Repetition:          {st['nRepetition']}")
        print(f"    - Learning Rate:                 {st['learningRate']}")
        print(f"    - Delta Ratio:                   {st['deltaRatio']}")
        print(f"    - Velocity Beta:                 {st['beta_velocity']}")
        print(f"    - Momentum Beta:                 {st['beta_momentum']}")
        print(f"    - Repopulation Ratio:            {st['repopulationRatio']}")
        print(f"    - Repopulation Interval:         {st['repopulationInterval']}")
        print(f"    - Repopulation Guide Ratio:      {st['repopulationGuideRatio']}")
        print(f"    - Repopulation Decay Rate:       {st['repopulationDecayRate']}")
        print(f"    - Scoring:                       {st['scoring']}")
        print(f"    - Scoring Maximum MDD:           {st['scoring_maxMDD']}")
        print(f"    - Scoring Growth Rate Weight:    {st['scoring_growthRateWeight']}")
        print(f"    - Scoring Growth Rate Scaler:    {st['scoring_growthRateScaler']}")
        print(f"    - Scoring Volatility Weight:     {st['scoring_volatilityWeight']}")
        print(f"    - Scoring Volatility Scaler:     {st['scoring_volatilityScaler']}")
        print(f"    - Scoring nTrades Weight:        {st['scoring_nTradesWeight']}")
        print(f"    - Scoring nTrades Scaler:        {st['scoring_nTradesScaler']}")
        print(f"    - Termination Threshold:         {st['terminationThreshold']}")

        #[4-7]: Seeker Best Result
        pResult_br = pResult['bestResult']
        print(f"    - Seeker Best Result:")
        print(f"      - Trade Parameters: {tuple(pResult_br['tradeParams'])}")
        print(f"      - Model Parameters: {tuple(pResult_br['modelParams'])}")
        balance_final        = pResult_br['finalBalance']
        balance_final_growth = balance_final/st['balance_initial']-1
        if   balance_final_growth < 0:  print(f"      - Final Balance:    {balance_final:.8f}", termcolor.colored(f"[{balance_final_growth*100:.3f} %]", 'light_red'))
        elif balance_final_growth == 0: print(f"      - Final Balance:    {balance_final:.8f}", termcolor.colored(f"[{balance_final_growth*100:.3f} %]", None))
        else:                           print(f"      - Final Balance:    {balance_final:.8f}", termcolor.colored(f"[+{balance_final_growth*100:.3f} %]", 'light_green'))
        growthRate_interval = pResult_br['growthRate']
        growthRate_daily    = math.exp(growthRate_interval*1440)        -1
        growthRate_monthly  = math.exp(growthRate_interval*1440*30.4167)-1
        volatility = pResult_br['volatility']
        volatility_tMin_997 = math.exp(-volatility*3)-1
        volatility_tMax_997 = math.exp( volatility*3)-1
        if   (growthRate_daily < 0):  print(f"      - Growth Rate:      {growthRate_interval:.8f} / {termcolor.colored(f'{growthRate_daily*100:.3f} %', 'light_red')} [Daily] / {termcolor.colored(f'{growthRate_monthly*100:.3f} %', 'light_red')} [Monthly]")
        elif (growthRate_daily == 0): print(f"      - Growth Rate:      {growthRate_interval:.8f} / {termcolor.colored(f'{growthRate_daily*100:.3f} %', None)} [Daily] / {termcolor.colored(f'{growthRate_monthly*100:.3f} %', None)} [Monthly]")
        else:                         print(f"      - Growth Rate:      {growthRate_interval:.8f} / {termcolor.colored(f'+{growthRate_daily*100:.3f} %', 'light_green')} [Daily] / {termcolor.colored(f'+{growthRate_monthly*100:.3f} %', 'light_green')} [Monthly]")
        print(f"      - Volatility:       {volatility:.8f} [Theoretical 99.7%: {termcolor.colored(f'{volatility_tMin_997*100:.3f} %', 'light_magenta')} / {termcolor.colored(f'+{volatility_tMax_997*100:.3f} %', 'light_blue')}]")
        print(f"      - Score:            {pResult_br['score']:.8f}")
        print(f"      - nTrades:          {pResult_br['nTrades']}")

        #[4-8]: eFunction Initialization
        eFunction = exitFunction(modelName              = st['exitFunctionType'],
                                 isSeeker               = False, 
                                 balance_initial        = st['balance_initial'],
                                 balance_allocation_max = st['balance_allocation_max'],
                                 leverage               = st['leverage'],
                                 isolated               = st['isolated'],
                                 pslReentry             = st['pslReentry'],
                                 precision_price        = descriptor['pricePrecision'],
                                 precision_quantity     = descriptor['quantityPrecision'],
                                 precision_quote        = descriptor['quotePrecision'],
                                 lmTable                = LEVERAGEMARGINTABLE.get(descriptor['positionSymbol'], None))
        print(f"    - Preprocessing Analysis Data...")
        t_0 = time.perf_counter_ns()
        ppResult = eFunction.preprocessData(linearizedAnalysis = linearizedAnalysis, indexIdentifier = descriptor['indexIdentifier'])
        if ppResult is None:
            print(f"    - Analysis Data Preprocessing Failed. Seeker Records Reconstruction Not Possible.")
            continue
        t_1 = time.perf_counter_ns()
        validityRate, gapRate = ppResult
        print( "    - Analysis Data Preprocessing Complete!")
        print(f"      - Elapsed Time:  {(t_1-t_0)/1e6:.3f} ms")
        print(f"      - Validity Rate: {validityRate*100:.3f} %")
        print(f"      - Gap Rate:      {gapRate*100:.3f} %")

        #[4-9]: eFunction Processing
        print(f"    - Reconstructing Seeker Records...")
        t_0 = time.perf_counter_ns()
        recs    = sorted(pResult['records'], key = lambda x: x['score'], reverse = True)[:100]
        params  = [rec['tradeParams']+rec['modelParams'] for rec in recs]
        nParams = len(params)
        (
        balance_wallet_history, 
        balance_margin_history, 
        balance_bestFit_history,
        balance_bestFit_growthRates,
        balance_bestFit_volatilities,
        nTrades,
        ) = eFunction.performOnParams(params = params)
        t_1 = time.perf_counter_ns()
        print(f"    - Seeker Records Reconstruction Complete! <{(t_1-t_0)/1e6:.3f} ms>")

        #[4-10]: Matplot Drawing
        #---[4-10-1]: Prepare Subplots
        fig, (ax1, ax2, ax3, ax4) = matplotlib.pyplot.subplots(4, 1, figsize=(16, 12), sharex=True)
        fig.subplots_adjust(hspace=0.2)

        #---[4-10-2]: Prepare Data
        idx_close = descriptor['indexIdentifier']['KLINE_CLOSEPRICE']
        hist_len = balance_wallet_history.shape[1]
        close_p  = linearizedAnalysis[-hist_len:, idx_close]
        bwh  = balance_wallet_history.cpu()
        mwh  = balance_margin_history.cpu()
        bbfh = balance_bestFit_history.cpu()

        #---[4-10-3]: Price Subplot
        ax1.plot(close_p, color='#1f77b4', linewidth=1.2, label='Close Price')
        ax1.set_title("Price (Close Price)")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=8)

        #---[4-10-4]: Balance History (Linear Scale)
        ax2.plot(bwh[0,:],  color=(0.0, 0.7, 1.0, 1.0), linestyle='solid',  linewidth=1, zorder=4, label='Wallet (Best)')
        ax2.plot(mwh[0,:],  color=(0.0, 0.7, 1.0, 0.5), linestyle='dashed', linewidth=1, zorder=3, label='Margin (Best)')
        ax2.plot(bbfh[0,:], color=(0.8, 0.5, 0.8, 1.0), linestyle='solid',  linewidth=2, zorder=5, label='Best Fit')
        for _rIndex in range(1, nParams):
            ax2.plot(bwh[_rIndex,:], color=(0.5, 0.8, 0.0, round((_rIndex+1)/nParams*0.10+0.10,3)), linestyle='solid',  linewidth=0.5, zorder=2)
            ax2.plot(mwh[_rIndex,:], color=(0.5, 0.8, 0.0, round((_rIndex+1)/nParams*0.05+0.05,3)), linestyle='dashed', linewidth=0.5, zorder=1)
        ax2.set_title("Balance History (Linear Scaled)")
        ax2.set_ylabel("Balance")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=8)

        #---[4-10-5]: Balance History (Log Scaled)
        ax3.plot(bwh[0,:],  color=(0.0, 0.7, 1.0, 1.0), linestyle='solid',  linewidth=1, zorder=4, label='Wallet (Best)')
        ax3.plot(mwh[0,:],  color=(0.0, 0.7, 1.0, 0.5), linestyle='dashed', linewidth=1, zorder=3, label='Margin (Best)')
        ax3.plot(bbfh[0,:], color=(0.8, 0.5, 0.8, 1.0), linestyle='solid',  linewidth=2, zorder=5, label='Best Fit')
        for _rIndex in range(1, nParams):
            ax3.plot(bwh[_rIndex,:], color=(0.5, 0.8, 0.0, round((_rIndex+1)/nParams*0.10+0.10,3)), linestyle='solid',  linewidth=0.5, zorder=2)
            ax3.plot(mwh[_rIndex,:], color=(0.5, 0.8, 0.0, round((_rIndex+1)/nParams*0.05+0.05,3)), linestyle='dashed', linewidth=0.5, zorder=1)
        ax3.set_yscale('log')
        ax3.set_title("Balance History (Log Scaled)")
        ax3.set_ylabel("Log Balance")
        ax3.set_xlabel("Time Step")
        ax3.grid(True, alpha=0.3, which='both', linestyle='--')
        ax3.legend(loc='upper left', fontsize=8)

        #---[4-10-6]: Balance Deviation from Best Fit (%)
        wallet_0  = bwh[0,:]
        margin_0  = mwh[0,:]
        bestFit_0 = bbfh[0,:]
        dev_wallet = (wallet_0 - bestFit_0) / bestFit_0 * 100
        dev_margin = (margin_0 - bestFit_0) / bestFit_0 * 100
        ax4.plot(dev_wallet, color=(0.0, 0.7, 1.0, 1.0), linestyle='solid',  linewidth=1.5, zorder=4, label='Wallet Deviation (%)')
        ax4.plot(dev_margin, color=(0.0, 0.7, 1.0, 0.5), linestyle='dashed', linewidth=1.5, zorder=3, label='Margin Deviation (%)')
        ax4.axhline(0, color=(0.8, 0.5, 0.8, 1.0), linestyle='solid', linewidth=2, zorder=5, label='Best Fit (0%)')
        ax4.set_title("Balance Deviation from Best Fit (%)")
        ax4.set_ylabel("Deviation (%)")
        ax4.set_xlabel("Time Step")
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(loc='upper left', fontsize=8)

        #---[4-10-7]: Matplot Show
        fig.suptitle(f"[RESULT READ] Analysis Results", fontsize=18, fontweight='bold')
        matplotlib.pyplot.tight_layout(rect=[0, 0.03, 1, 0.96])
        matplotlib.pyplot.show()

        #[4-11]: Line Skip
        print()

    #[5]: System Message
    print(termcolor.colored("[Seeker Result Read Complete]", 'light_blue'))
#READ FUNCTION END ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------










#MAIN FUNCTION ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    #[1]: System Message
    print(termcolor.colored("<TARGET EXPOSURE FACTOR FUNCTION PARAMETERS SEEKER PROCESS>\n", 'light_green'))
    process_begin_time = int(time.time())

    #[2]: Data Folders Setup
    path_folder_analysisData = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')
    path_folder_results      = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    if not os.path.isdir(path_folder_analysisData): os.mkdir(path_folder_analysisData)
    if not os.path.isdir(path_folder_results):      os.mkdir(path_folder_results)

    #[3]: Configuration Read
    config_test = None
    config_seek = None
    rCord_read  = None
    mode        = config.MODE
    if   mode == 'TEST': config_test = config.PARAMETERTEST
    elif mode == 'SEEK': config_seek = config.SEEKERTARGETS
    elif mode == 'READ': rCord_read  = config.RCODETOREAD

    #[4]: Parameter Test
    if config_test is not None:
        test(config_test = config_test)
        
    #[5]: Seeker
    if config_seek is not None:
        rCord_read = seek(config_seek = config_seek,
                          process_begin_time = process_begin_time)

    #[6]: Read Data
    if rCord_read is not None:
        read(rCord_read = rCord_read)

    #[7]: System Message
    print(termcolor.colored("\n<TARGET EXPOSURE FACTOR FUNCTION PARAMETERS SEEKER PROCESS COMPLETE>", 'light_green'))
#MAIN FUNCTION END ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------