from typing import Tuple, Sequence, Dict, Any
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from scoop import futures
from deap import creator, base, tools
import jenkspy

LAND_USES = {
    0: 'border',
    1: 'agriculture',
    2: 'conservation',
    3: 'urban',
    4: 'water',
    5: 'herbaceous'
}

LAND_USES_ABBR = {
    'border': 'border',
    'agriculture': 'ag',
    'conservation': 'con',
    'urban': 'urb',
    'water': 'water',
    'herbaceous': 'herb',
}

# --- variables need to run GA (read from user input) ---
base_df = pd.read_csv('data/base_df.csv', index_col=0)
GRID_R = 198
GRID_C = 196

TEMP_PER_AG = 0.0038
TEMP_PER_CON = -0.0020
TEMP_PER_URB = 0.0103
TEMP_PER_WATER = 0
TEMP_PER_HERB = 0.0072
TEMP_TARGET = 2

PPL_PER_URB = 514
PPL_CURRENT = 4157733
PPL_GROWTH = 1000000

LU_INSUFFICIENT = (3,)
LU_EXCESS = (5,)

min_conf = 0
max_conf = GRID_C * GRID_R

# --- Column names for the input DataFrame ---
AG_SUIT = 'ag_suit'  # agriculture suitability
CON_SUIT = 'con_suit'  # conservation suitability
URB_SUIT = 'urb_suit'  # urban suitability
CURRENT_LU = 'current_lu'  # current land use

# --- GA parameters ---
CXPB = 0.6  # crossover probability
MUTPB = 0.5  # mutation probability
GEN_NUM = 100  # number of generations
POP_SIZE = 100  # population size

# --- Initial sampling probability for each land use ---
INIT_PROB_AG = {1: 0.1, 2: 0.2, 3: 0.7}
INIT_PROB_CON = {1: 0.1, 2: 0.1, 3: 0.8}
INIT_PROB_URB = {1: 0.1, 2: 0.3, 3: 0.6}
INIT_SAMPLE_SIZE = 3000  # number of cells changing its land use


class LandUse:
    """Land use class for GA analysis."""

    def __init__(self, landuse):
        try:
            self.lu_name = LAND_USES[landuse]
            self.lu_abbr = LAND_USES_ABBR[self.lu_name]
            self.lu_code = landuse
        except KeyError:
            try:
                self.lu_abbr = LAND_USES_ABBR[landuse]
                self.lu_code = list(LAND_USES.values()).index(landuse)
                self.lu_name = landuse
            except KeyError:
                try:
                    self.lu_code = list(LAND_USES_ABBR.values()).index(landuse)
                    self.lu_name = LAND_USES[self.lu_code]
                    self.lu_abbr = LAND_USES_ABBR[self.lu_name]
                except ValueError:
                    raise ValueError("Invalid land use.")

    def __str__(self):
        return self.lu_name

    def is_border(self):
        return self.lu_code == 0

    def is_water(self):
        return self.lu_code == 4

    def is_herb(self):
        return self.lu_code == 5

    def is_core(self):
        if self.is_herb() or self.is_water() or self.is_border():
            return False
        return True


class LandUsePreference:
    def __init__(self, lu_code: int, preference: int):
        self.landuse = lu_code
        self.preference = preference

    @staticmethod
    def check_landuse(value):
        if value not in (1, 2, 3):
            raise ValueError('Only ag, con, and urb can have a preference.')
        return LandUse(value)

    @staticmethod
    def check_preference(value):
        if value not in (1, 2, 3):
            raise ValueError('Preference must be 1, 2, or 3.')
        return value

    @property
    def landuse(self):
        return self._landuse

    @landuse.setter
    def landuse(self, value):
        self._landuse = self.check_landuse(value)

    @property
    def preference(self):
        return self._preference

    @preference.setter
    def preference(self, value):
        self._preference = self.check_preference(value)


class LandUseConflict:
    def __init__(self, ag_pref, con_pref, urb_pref):
        self.ag_pref = LandUsePreference(1, ag_pref)    # ag: lu_code=1
        self.con_pref = LandUsePreference(2, con_pref)  # con: lu_code=2
        self.urb_pref = LandUsePreference(3, urb_pref)  # urb: lu_code=3
        self.preferences = [self.ag_pref.preference,
                            self.con_pref.preference,
                            self.urb_pref.preference]
        self.conflict_freq = None
        self.conflict_uses = None

    @property
    def preferred_uses(self) -> Tuple:
        preferred_val = max(self.preferences)
        return tuple(
            [k for k, v in self.as_dict().items() if v == preferred_val]
        )

    @property
    def isinconflict(self) -> bool:
        if len(set(self.preferences)) != 3:  # if there's repetitive preference
            return True
        else:
            return False

    def calculate(self) -> float:
        if self.isinconflict:
            # most occurrences preference value reveals land-use conflict
            conflict_val = max(self.preferences, key=self.preferences.count)
            self.conflict_uses = tuple(
                [k for k, v in self.as_dict().items() if v == conflict_val]
            )
            # count the number of occurrences with the conflict value
            self.conflict_freq = self.preferences.count(conflict_val)
            return (
                self.conflict_freq *
                np.sqrt(
                    np.einsum(
                        'i,i->',
                        self.preferences,
                        self.preferences
                    )
                )
            )
        else:
            self.conflict_uses = tuple()
            self.conflict_freq = 0
            return 0

    def as_dict(self) -> Dict:
        return dict(zip([1, 2, 3], self.preferences))

    def urban_conflict(self) -> float:
        lup_dict = self.as_dict()
        urb_ag = lup_dict[3] / lup_dict[1]
        urb_con = lup_dict[3] / lup_dict[2]
        if urb_ag > 1:
            urb_ag_conf = 0
        else:
            urb_ag_conf = 1 / urb_ag
        if urb_con > 1:
            urb_con_conf = 0
        else:
            urb_con_conf = 1 / urb_con
        return urb_ag_conf + urb_con_conf


class Neighborhood:
    def __init__(self, lu_codes: Tuple[int]):
        if len(lu_codes) != 9:
            raise ValueError("Neighborhood must have 9 cells in the array.")
        self.landuses = np.array([LandUse(_) for _ in lu_codes])
        self.neighborhood_lu_codes = np.array(lu_codes)

    @property
    def center_cell_lu(self):
        center_cell_lu = self.landuses[4]
        if center_cell_lu.is_water() or center_cell_lu.is_border():
            raise ValueError("Center cell land use can't be water or border.")
        return center_cell_lu

    @property
    def compactness(self):
        if self.center_cell_lu.is_herb():
            return 1
        # change water to border
        self.neighborhood_lu_codes[self.neighborhood_lu_codes == 4] = 0
        n_same_as_center = np.sum(
            self.neighborhood_lu_codes == self.center_cell_lu.lu_code
        )
        n_valid_neighbor = np.count_nonzero(self.neighborhood_lu_codes)
        return (n_same_as_center - 1)/(n_valid_neighbor - 1)


class Scenario:
    def __init__(self, lu_array):
        # this array must contain all cells in a rectangular shape
        self.lu_array = lu_array
        self.n_row, self.n_col = lu_array.shape
        self.mask = np.isin(self.lu_array, [1, 2, 3, 5])
        self.water_indices = np.where(self.lu_array == 4)

    @property
    def n_ag(self):
        return np.count_nonzero(np.isin(self.lu_array, 1))

    @property
    def n_con(self):
        return np.count_nonzero(np.isin(self.lu_array, 2))

    @property
    def n_urb(self):
        return np.count_nonzero(np.isin(self.lu_array, 3))

    @property
    def water_indices_1d(self):
        return self.water_indices[0]*self.n_col + self.water_indices[1]

    def _neighborhoods(self):
        neighborhood_shape = (3, 3)  # a 3x3 neighborhood
        padded_scenario = np.pad(self.lu_array,
                                 ((1, 1), (1, 1)),
                                 'constant',
                                 constant_values=0)
        strides = padded_scenario.strides + padded_scenario.strides
        neighborhoods = np.lib.stride_tricks.as_strided(
            padded_scenario,
            shape=((padded_scenario.shape[0] - neighborhood_shape[0] + 1,) +
                   (padded_scenario.shape[1] - neighborhood_shape[1] + 1,) +
                   neighborhood_shape),
            strides=strides
        ).reshape(self.lu_array.size, 9)
        neighborhoods_no_border_water = neighborhoods[
            self.mask.reshape(-1)
        ]
        return np.apply_along_axis(
            Neighborhood,
            axis=1,
            arr=neighborhoods_no_border_water
        )

    def total_compactness(self):
        return np.sum([_.compactness for _ in self._neighborhoods()])

    def plot(self, data=None, ax=None):
        cmap = colors.ListedColormap(
            ['#FF000000', '#7FFF82', '#064808', '#B40300', 'blue', '#A66907']
        )
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6], cmap.N)
        if data is None:
            data = self.lu_array
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            return ax.pcolor(data[::-1], cmap=cmap, norm=norm,
                             edgecolors='k', linewidths=0.5)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            return ax.pcolor(data[::-1], cmap=cmap, norm=norm,
                             edgecolors='k', linewidths=0.5)


def cx_grid(ind1, ind2, n_row=30, n_col=30, indpb=0.5):
    """customize crossover. crossover between 3x3 grid in ind1 and ind2.
    iff the center cell's land use is the same with at least one of the
    neighbors in the other individual's center cell.
    """
    cx_size = round(indpb * n_row * n_col)
    cx_position_1d = np.random.choice(len(ind1), cx_size, replace=False)
    cx_ir = cx_position_1d // n_row
    cx_ic = cx_position_1d % n_col
    # cx_ir = [1, 3, 1, 4]  # example row index
    # cx_ic = [1, 4, 0, 0]  # example column index
    ind1 = ind1.reshape((n_row, n_col))
    ind2 = ind2.reshape((n_row, n_col))
    ind1_padded = np.pad(ind1, ((1, 1), (1, 1)),
                         'constant', constant_values=0)
    ind2_padded = np.pad(ind2, ((1, 1), (1, 1)),
                         'constant', constant_values=0)
    for (r, c) in zip(cx_ir, cx_ic):
        ind1_change = False
        ind2_change = False
        if ind1[r, c] in ind2_padded[r:r+3, c:c+3]:
            ind2_change = ind1[r, c]
        if ind2[r, c] in ind1_padded[r:r+3, c:c+3]:
            ind1_change = ind2[r, c]
        if ind1_change:
            ind1[r, c] = ind1_change
        if ind2_change:
            ind2[r, c] = ind2_change
    return ind1.reshape(-1), ind2.reshape(-1)


def mut_grid(ind, n_row=30, n_col=30, lu_in=(3,), lu_ex=(1,), indpb=0.1):
    """customize mutation. randomly select cells lu_in (inadequate land uses)
    with probability of indpb. convert selected and the surrounding neighbors
    (3x3 grid) to a land use randomly selected from lu_ex (excessive land uses)
    """
    mut_size = round(indpb * n_row * n_col)
    lu_ex_i = np.where(np.isin(ind, lu_ex))[0]
    if mut_size > len(lu_ex_i):
        mut_size = len(lu_ex_i)
    mut_i = np.random.choice(lu_ex_i, mut_size)
    mut_ir = mut_i // n_row
    mut_ic = mut_i % n_col
    # mut_ir = [1, 3]  # example row index
    # mut_ic = [1, 4]  # example column index

    ind_padded = np.pad(ind.reshape((n_row, n_col)),
                        ((1, 1), (1, 1)),
                        'constant', constant_values=0)
    for (r, c) in zip(mut_ir, mut_ic):
        ind_padded[r:r+3, c:c+3] = np.random.choice(lu_in, 1)[0]

    ind[:] = ind_padded[1:n_row+1, 1:n_col+1].reshape(-1)
    return ind,


# -------------------------Initialization---------------------- #
def nb_classify(sr, num_class=3):
    breaks = jenkspy.jenks_breaks(sr, n_classes=num_class)
    return pd.Series(
        pd.cut(
            sr,
            bins=breaks,
            labels=np.arange(1, num_class+1),
            include_lowest=True
        ),
        dtype='int'
    )


def random_scn(df, landuse, lu_col, n=3000):
    if landuse == 0:  # copy the original scenario
        return df[lu_col].values
    if n > len(df) * 0.5:
        raise ValueError('n must be less than half of the dataframe size.')
    lu = LandUse(landuse)
    if lu.lu_code == 1:
        prob_dict = INIT_PROB_AG
    elif lu.lu_code == 2:
        prob_dict = INIT_PROB_CON
    elif lu.lu_code == 3:
        prob_dict = INIT_PROB_URB
    else:
        raise ValueError('landuse must be 1 (ag), 2 (con), or 3 (urb).')
    suit_col = lu.lu_abbr + '_suit'
    nbc_sr = nb_classify(df.loc[df[lu_col] != lu.lu_code, suit_col])
    prob_sr = pd.Series(prob_dict) / nbc_sr.value_counts()

    lu_sample = np.random.choice(
        nbc_sr.index.values,
        size=n,  # this can be changed depending on size of the scenario
        replace=False,
        p=nbc_sr.map(prob_sr)
    )
    scenario = df[lu_col].copy()
    scenario.iloc[lu_sample] = lu.lu_code
    return scenario.values


init_random_scn = partial(
    random_scn,
    base_df,
    np.random.choice(4, p=[0.4, 0.2, 0.2, 0.2]),
    CURRENT_LU,
    INIT_SAMPLE_SIZE
)


def evaluate(individual):
    individual = individual[0]
    scn = Scenario(individual.reshape(GRID_R, GRID_C))
    p = 4  # penalty coefficient
    grid_size = GRID_R * GRID_C

    # ------------------------ additive objective ------------------------ #
    # 1. conflict: minimize conflict due to allocating urban cells
    allocate_conf = np.sum(base_df['urban_conflict'][individual == 3])
    # minimize conflict
    f_conf = (allocate_conf - min_conf) / (max_conf - min_conf)

    # 2. suitability: maximize suitability for ag, con, and urb
    min_suit = grid_size
    max_suit = 9 * grid_size
    allocate_suit = (
            base_df.loc[individual == 1, AG_SUIT].sum() +
            base_df.loc[individual == 2, CON_SUIT].sum() +
            base_df.loc[individual == 3, URB_SUIT].sum()
    )
    # maximize suitability
    f_suit = (allocate_suit - max_suit) / (min_suit - max_suit)

    # -------------------------spatial objective------------------------- #
    min_comp = 0
    max_comp = grid_size
    allocate_compact = scn.total_compactness()
    # maximize compactness
    f_spa = (allocate_compact - max_comp) / (min_comp - max_comp)

    # --------------------------constraints------------------------------ #
    # 1. Temperature: calculate temperature difference
    temp_slope = pd.Series(
        (TEMP_PER_AG,
         TEMP_PER_CON,
         TEMP_PER_URB,
         TEMP_PER_WATER,
         TEMP_PER_HERB),
        index=np.arange(1, 6)
    )
    num_lu_diff = (
            pd.Series(individual).value_counts() -
            base_df[CURRENT_LU].value_counts()
    )
    temp_diff = num_lu_diff @ temp_slope  # dot product
    if temp_diff == 0:
        temp_constraint = 0
    else:
        temp_diff_normalized = (temp_diff - TEMP_TARGET)/abs(temp_diff)
        temp_constraint = max([0, temp_diff_normalized])**2

    # 2. Population: number of urban cells
    urb_cell = np.sum(individual == 3)
    urb_cell_needed = int(np.ceil((PPL_GROWTH + PPL_CURRENT)/PPL_PER_URB))
    urb_diff_normalized = (urb_cell_needed - urb_cell)/urb_cell_needed
    pop_constraint = max([0, urb_diff_normalized])**2
    return (
        (f_conf**p)*(f_spa**p) +
        (f_suit**p)*(f_spa**p) +
        temp_constraint +
        pop_constraint,
    )


# -------------------------GA configurations---------------------------- #
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("map", futures.map)

toolbox.register(
    "individual", tools.initRepeat, creator.Individual, init_random_scn, n=1
)
toolbox.register(
    "population", tools.initRepeat, list, toolbox.individual
)
toolbox.register("evaluate", evaluate)

toolbox.register(
    "mate", cx_grid,
    n_row=GRID_R, n_col=GRID_C,
    indpb=0.5
)
toolbox.register(
    "mutate", mut_grid,
    n_row=GRID_R, n_col=GRID_C,
    lu_in=LU_INSUFFICIENT,
    lu_ex=LU_EXCESS,
    indpb=0.1
)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    hof.clear()
    # Evaluate the entire population
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # base scenario
    base_scn = Scenario(base_df[CURRENT_LU].values.reshape(GRID_R, GRID_C))

    g = 0

    while g < GEN_NUM:
        g += 1
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals to avoid modifying the original
        # this uses the `copy.deepcopy()` function
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < CXPB:
                toolbox.mate(child1[0], child2[0])
                # set water cells back to water
                child1[0][base_scn.water_indices_1d] = 4
                child2[0][base_scn.water_indices_1d] = 4
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < MUTPB:
                toolbox.mutate(mutant[0])
                # set water cells back to water
                mutant[0][base_scn.water_indices_1d] = 4
                del mutant.fitness.values

        # recalculate fitness for (crossed or mutated) individuals
        # i.e., those with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(offspring)

        fits = [ind.fitness.values[0] for ind in pop]

        if g % 10 == 0:
            print(f'gen {g}: {min(fits)}, {max(fits)}, {np.mean(fits)}')

    return hof[0]
