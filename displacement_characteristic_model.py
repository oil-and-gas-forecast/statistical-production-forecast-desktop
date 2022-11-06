import numpy as np


class DisplacementCharacteristic:

    def __init__(
        self,
        oil_production,
        liq_production,
        niz,
        considerations,
        well_name,
        mark,
        wc_fact,
        rf_now
    ):
        self.oil_production = oil_production
        self.liq_production = liq_production
        self.niz = niz
        self.considerations = considerations
        self.well_name = well_name
        self.mark = mark
        self.wc_fact = wc_fact
        self.rf_now = rf_now

    def solver(
        self,
        correlation_coeffs
    ):
        corey_oil, corey_water, mef = correlation_coeffs

        v1 = np.cumsum(self.oil_production) / self.niz / 1e3
        v1 = np.delete(v1, 0)
        v1 = np.insert(v1, 0, 0)
        k1 = (1 - v1) ** corey_oil / ((1 - v1) ** corey_oil + mef * v1 * corey_water)

        v2 = v1 + self.liq_production * k1 / 2 / self.niz / 1e3
        k2 = (1 - v2) ** corey_oil / ((1 - v2) ** corey_oil + mef * v2 * corey_water)

        v3 = v1 + self.liq_production * k2 / 2 / self.niz / 1e3
        k3 = (1 - v3) ** corey_oil / ((1 - v3) ** corey_oil + mef * v3 * corey_water)

        v4 = v1 + self.liq_production * k3 / 2 / self.niz / 1e3
        k4 = (1 - v4) ** corey_oil / ((1 - v4) ** corey_oil + mef * v4 * corey_water)

        oil_production_model = self.liq_production / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        oil_production_model[oil_production_model == -np.inf] = 0
        oil_production_model[oil_production_model == np.inf] = 0

        deviation = [(oil_production_model - self.oil_production) ** 2]

        return np.sum(deviation)

    def to_conditions(
        self,
        correlation_coeffs
    ):
        corey_oil, corey_water, mef = correlation_coeffs
        point = self.considerations[self.well_name][0]

        if np.isnan(point) or self.mark is False:
            point = 1
        if point == 1:
            if self.wc_fact.size > 1:
                wc_last = self.wc_fact[-1]
            else:
                wc_last = self.wc_fact
        elif point == 3:
            if self.wc_fact.size >= 3:
                wc_last = np.average(self.wc_fact[-3:-1])
            elif self.wc_fact.size == 2:
                wc_last = np.average(self.wc_fact[-2:-1])
            else:
                wc_last = self.wc_fact
        else:
            print('Неверный формат для условия привязки! Привязка будет осуществляться к последней точке')
            if self.wc_fact.size > 1:
                wc_last = self.wc_fact[-1]
            else:
                wc_last = self.wc_fact

        wc_model = mef * self.rf_now ** corey_water / \
                   ((1 - self.rf_now) ** corey_oil + mef * self.rf_now ** corey_water)
        binding = wc_model - wc_last

        return binding
