import json
import pandas as pd


def addendum(actual, sat, phase, direction, dFrame):
    row = []
    row.append(phase)
    row.append(direction+'bound')
    row.append(int(actual[direction]))
    # eqHourly = round(int(actual[direction])/float(actual['PHF']), 2)
    # row.append(eqHourly)
    totEqHourly = round(int(actual[direction])/float(actual['PHF']), 2)
    row.append(totEqHourly)
    row.append(int(sat[direction]))
    row.append(round(totEqHourly/int(sat[direction]), 2))
    Yi = round(totEqHourly/int(sat[direction]), 2)
    row.append(Yi)
    dFrame.loc[len(dFrame.index)] = row
    return dFrame


if __name__ == '__main__':
    with open('flowClassified.json') as fin:
        flow = json.load(fin)
    init = {
            '1': ['Phase'],
            '2': ['Lane Group'],
            '3': ['Hourly Volume'],
            '4': ['Total Eq. Hourly Volume'],
            '5': ['Sat. Flow'],
            '6': ['Qij/Sj'],
            '7': ['Yi']
            }
    df0 = pd.DataFrame(init)

    df0 = addendum(flow['intersection1'], flow['saturationFlow'], 'A', 'north', df0)
    df0 = addendum(flow['intersection1'], flow['saturationFlow'], 'B', 'south', df0)
    df0 = addendum(flow['intersection1'], flow['saturationFlow'], 'C', 'east', df0)
    df0 = addendum(flow['intersection1'], flow['saturationFlow'], 'D', 'west', df0)

    temp = df0['7']
    Y = temp[1:].sum()
    # print(Y)
    # Total lost time
    L = 4*(float(flow['intersection1']['totalLostTime'])+float(flow['intersection1']['allRed']))
    # print(L)
    # Optimum cycle length
    C = (1.5*L+5)/(1-Y)
    # print(C)
    # Total effective green time
    gTot = C-L
    # print(gTot)
    # Effective green times
    gA = temp[1]/Y*gTot
    gB = temp[2]/Y*gTot
    gC = temp[3]/Y*gTot
    gD = temp[4]/Y*gTot
    # print(gD)
    g = (gA+gB+gC+gD)/4
    # Actual green time
    GA = gA+float(flow['intersection1']['totalLostTime'])-float(flow['intersection1']['yellow'])
    GB = gB+float(flow['intersection1']['totalLostTime'])-float(flow['intersection1']['yellow'])
    GC = gC+float(flow['intersection1']['totalLostTime'])-float(flow['intersection1']['yellow'])
    GD = gD+float(flow['intersection1']['totalLostTime'])-float(flow['intersection1']['yellow'])
    # print(GA)
    # Check
    if (GA+GB+GC+GD)+4*float(flow['intersection1']['yellow'])+4*float(flow['intersection1']['allRed']) <= C:
        print('Cycle Length Calculation is Satisfactory')
    print(df0)

    # Calculation of delay and demarcation of LOS
    T = C/4
    v = 0.8       # departure rate = 5 veh/s, calculated from above data explicitly
    cap = 0.5  # capacity of lane = 1800 veh/hr/ln = 0.5 veh/s/ln
    AD = round((C-g)/2 + T/2*(v/cap-1), 2)
    print('\nAverage Delay per vechile: ', AD)
    with open('HCM_LOS.json') as fin:
        los = json.load(fin)
    los = los['type1']
    if AD < float(los['A']):
        LOS = 'Grade A'
    elif AD < float(los['B']):
        LOS = 'Grade B'
    elif AD < float(los['C']):
        LOS = 'Grade C'
    elif AD < float(los['D']):
        LOS = 'Grade D'
    elif AD < float(los['E']):
        LOS = 'Grade E'
    elif AD > float(los['F']):
        LOS = 'Grade F'
    print('Level of Service of the intersection: ', LOS)