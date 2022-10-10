import subprocess
import pkg_resources
import sys
from random import randint

# ensure that the required packages are installed and install otherwise.
required = {'pygame', 'numpy'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

import numpy as np
import pygame

POPULATION = 100
MUTATION_PROB = 10
CROSSOVER_PROB = 10

class Gui:
    # This class handles the gui aspects of this program.
    pygame.display.set_caption('Futoshiki Solver')
    pause = False
    block_size = 40
    background_color = (255, 255, 255)

    def __init__(self, size):
        self.size = size
        pygame.init()
        self.TitleFont = pygame.font.SysFont('comicsans', 30)
        self.TitleFont.underline = True
        self.font = pygame.font.SysFont('comicsans', 20)
        board_size = (self.width, self.height) = (self.size * self.block_size) + 260, (
                    self.size * self.block_size) + 260

        self.board = pygame.display.set_mode(board_size)
        self.start_coord = (self.width - (1.5*self.size*self.block_size - self.size))/2

    def draw_constraint(self, first, second):
        # it gets the coordinates of the constraints - where the inequality sign should be, and calculates the coords
        # for the three points from which we will draw a line between them, in order to create an inequality sign.
        x1, y1 = first
        x2, y2 = second

        if x1 == x2 and y1 == y2:  # if the two coords are identical, the input file has a mistake in it.
            return
        if x1 == x2:
            if y2 > y1:
                p1, p2, p3 = self.calculate_points(x1, y1, True, 2)  # 2 indicates inequality facing right side, y2>y1
            elif y1 > y2:
                p1, p2, p3 = self.calculate_points(x1, y2,True, 1)  # 1 indicates inequality facing left side, y1>y2

        else:
            if x2 > x1:
                p1, p2, p3 = self.calculate_points(x1, y1, False, 2)  # 2 indicates inequality facing down, x2>x1

            elif x1 > x2:
                p1, p2, p3 = self.calculate_points(x2, y1, False, 1)  # 1 indicates inequality facing up, x1>x2

        # draws the lines from p1 to p2 and from p3 to p2, creating the inequality signs.
        pygame.draw.line(self.board, (0, 0, 0,), p1, p2, 2)
        pygame.draw.line(self.board, (0, 0, 0,), p3, p2, 2)

    def calculate_points(self, x, y, arex1x2equal, biggerid):
        c = 5
        push = 50  # used to push everything we draw a bit downwards in the gui, to make space for the text
        halfblock = 0.5 * self.block_size
        oneandhalfblock = 1.5 * self.block_size
        addition1, addition2, addition3  = - c, -c, -c
        if biggerid == 2:
            addition1, addition3 = c - halfblock, c - halfblock
        else:
            addition2 = c - 0.5 * self.block_size
        if arex1x2equal:
            p1 = (self.start_coord + y * oneandhalfblock + addition1,
                self.start_coord + (x - 1) * oneandhalfblock + c + push)
            p2 = (self.start_coord + y * oneandhalfblock + addition2,
                self.start_coord + x * oneandhalfblock - self.block_size + push)
            p3 = (self.start_coord + y * oneandhalfblock + addition3,
                self.start_coord + x * oneandhalfblock - halfblock - c + push)
        else:
            p1 = (self.start_coord + (y - 1) * oneandhalfblock + c,
                  self.start_coord + x * oneandhalfblock + addition1 + push)
            p2 = (self.start_coord + y * oneandhalfblock - self.block_size,
                  self.start_coord + x * oneandhalfblock + addition2 + push)
            p3 = (self.start_coord + y * oneandhalfblock - halfblock - c,
                  self.start_coord + x * oneandhalfblock + addition3 + push)

        return p1, p2, p3

    def write_text(self, text, x, y):
        # writes the text in the given x and y coordinates.
        text_surface = self.font.render(text, True, (0, 0, 0,))
        text_rect = text_surface.get_rect()
        text_rect.center = (x, y)
        self.board.blit(text_surface, text_rect)

    def show_solution(self, solution, constraints, generation, best_fit, avg_fit, worst_fit):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.pause = True
                    self.gui_paused()

        solution = solution.astype(int)
        self.board.fill(self.background_color)
        # writes the text at the top of the gui
        self.write_text(f"Generation: {generation}", self.width/2, 20)
        self.write_text(f"Best fitness: {best_fit}", self.width/2, 45)
        self.write_text(f"Average fitness: {avg_fit}", self.width/2, 70)
        self.write_text(f"Worst fitness: {worst_fit}", self.width/2, 95)
        # draw the constraints
        for constraint in constraints:
            self.draw_constraint(constraint[0], constraint[1])
        # draw the squares and the numbers in them
        for i in range(self.size):
            for j in range(self.size):
                x = self.start_coord + i*1.5*self.block_size
                y = self.start_coord + j*1.5*self.block_size + 50
                rect_obj = pygame.draw.rect(self.board, (0, 0, 0), (x, y, self.block_size, self.block_size), 2)
                text_surface = self.font.render(str(solution[j][i]), True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=rect_obj.center)
                self.board.blit(text_surface, text_rect)
        pygame.display.flip()

    def gui_paused(self):
        # pause the screen until the space bar is pressed again
        while self.pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.pause = False


def get_input(path):
    with open(path, 'r') as f:
        # reading all lines from input file and stripping them into a list of strings
        lines = [line.rstrip() for line in f.readlines()]
        # extracting the size of the matrix and initializing it
        size = int(lines[0])
        matrix = np.empty((size, size))
        matrix[:] = np.nan
        # extracting the amount of numbers given and assigning them in the matrix
        nums_given = int(lines[1])
        indexes_of_num_given = []
        for i in range(2, 2 + nums_given):
            nums = [int(num) for num in lines[i].split(" ")]
            matrix[nums[0] - 1][nums[1] - 1] = nums[2]
            indexes_of_num_given.append((nums[0] - 1, nums[1] - 1))
        # saving in a list the coords of the "greater than" signs
        constraints = []
        for j in range(i + 2, len(lines)):
            x1, y1, x2, y2 = [int(num) for num in lines[j].split(" ")]
            constraints.append([(x1, y1), (x2, y2)])
        return size, matrix, indexes_of_num_given, constraints


def gen_random_solution(size, matrix, indexes):
    # creates an empty solution, and assigns a random number from 1 to the given size in each cell, except those with
    # given numbers in the input file. the function fills the matrix so that each row  will contain unique elements
    sol = np.copy(matrix)
    for i in range(size):
        for j in range(size):
            if (i, j) in indexes:
                continue
            val = randint(1, size)
            while val in sol[i, :]:
                val = randint(1, size)
            sol[i][j] = val
    return sol


def get_random_element(lst):
    # returns a random element in a list by drawing a random index from 0 to the list's size
    index = randint(0, len(lst)-1)
    return lst[index]


def is_constraint_maintained(constraint, sol):
    # checks if the constraints - the inequality signs - are met. if yes, it returns true. otherwise, false
    x1, y1 = constraint[0]
    x2, y2 = constraint[1]
    if sol[x1 - 1][y1 - 1] > sol[x2 - 1][y2 - 1]:
        return True
    else:
        return False


class GeneticAlg:
    def __init__(self, path):
        self.size, self.matrix, self.indexes, self.constraints = get_input(path)
        self.gen = self.first_gen()
        self.best_fit = float('-inf')
        self.avg_fit = 0.0
        self.worst_fit = float('inf')
        self.sols_fit = []
        self.index_best_sol = -1
        self.gui = Gui(self.size)
        self.mutation_prob = 10

    def first_gen(self):
        # generates random solutions in the size of POPULATION for the first gen
        gen = []
        for i in range(POPULATION):
            gen.append(gen_random_solution(self.size, self.matrix, self.indexes))
        return gen

    def calc_score(self, sol):
        score = 0
        # check if there's one appearance of every digit
        for i in range(1, self.size+1):
            # the next two lines count how many appearances of the digit i there are in each row and column
            row_is_unique = np.count_nonzero(np.count_nonzero(sol == i, axis=1))
            col_is_unique = np.count_nonzero(np.count_nonzero(sol == i, axis=0))

            # then, it checks if the amount of appearances of the digit i is equal to self.size, it means that there are
            # no repeats of it in the rows/columns, so we raise the score. otherwise, we lower it by the amount of
            # digits that are missing, so to "punish" more harshly if there are many repeats
            if row_is_unique == self.size:
                score += 700
            else:
                score -= 400 * (self.size - np.amax(np.count_nonzero(sol == i, axis=1)))
            if col_is_unique == self.size:
                score += 700
            else:
                score -= 400 * (self.size - np.amax(np.count_nonzero(sol == i, axis=0)))

        # check if the digits near the "greater than" signs maintain it, and adjust the score accordingly.
        for constraint in self.constraints:
            if is_constraint_maintained(constraint, sol):
                score += 800
            else:
                score -= 900

        return score

    def calc_fitness(self):
        # 'reset' the members for the new generation
        sols_fitness = []
        self.best_fit = float('-inf')
        self.worst_fit = float('inf')
        self.index_best_sol = -1
        sum_fitness = 0
        index = 0
        for sol in self.gen:
            # calculate the score for sol, and add it to sols_fit member
            sol_score = self.calc_score(sol)
            sols_fitness.append(sol_score)
            sum_fitness += sol_score
            # update the worst_fit and best_fit variables accordingly. if we update best_fit, we want to keep the index
            # for the best solution, so we also update that member
            if sol_score < self.worst_fit:
                self.worst_fit = sol_score
            if sol_score >= self.best_fit:
                self.best_fit = sol_score
                self.index_best_sol = index
            index += 1
        self.avg_fit = "%.2f" % (sum_fitness / POPULATION)
        return sols_fitness

    def fitness(self):
        return self.calc_fitness()

    def mutate_sol(self, sol):
        new_sol = np.empty((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in self.indexes:
                    # if the val in i, j is a number given by the input, we don't want to change it. so we assign the
                    # value and continue to the next iteration.
                    new_sol[i][j] = sol[i][j]
                    continue
                # pick a random integer. if it's below the mutation probability, we want to change the value of the num
                # in i, j to a value that is different from the one it currently has - to create a mutation.
                # if the random integer is above the mutation probability, we keep the old value.
                prob = randint(0, 100)
                if prob <= self.mutation_prob:
                    val = sol[i][j]
                    while val == sol[i][j]:
                        val = randint(1, self.size)
                    new_sol[i][j] = val
                else:
                    new_sol[i][j] = sol[i][j]
        return new_sol

    def crossover_sol(self, first_sol, second_sol):
        new_sol = np.empty(self.size * self.size)
        new_sol[:] = np.nan
        # randomly pick and integer that will dictate until which index we take the solution from the first solution,
        # and afterwards we take it from the second solution.
        index = randint(0, self.size * self.size - 1)
        # we use flatten to turn the solutions into 1d arrays, to make it easier to split them by the index we picked.
        new_sol[0:index] = first_sol.flatten()[0:index]
        new_sol[index:] = second_sol.flatten()[index:]
        return new_sol.reshape([self.size, self.size])

    def best_sols(self):
        # for the new generation, we want the best 20% solutions of the previous generation
        amount = int(POPULATION * 0.2)
        # we zip the sols_fit and gen into a list, sort it by the first element so that it sorts by the fitness of every
        # solution in a reverse order so the fitness is sorted in descending order, and then we extract the two
        # variables from the sorted zipped variable.
        self.sols_fit, self.gen = (list(t) for t in zip(*sorted(list(zip(self.sols_fit, self.gen)),
                                                                key=lambda x: x[0], reverse=True)))
        # return the best 20% solutions
        return self.gen[0:amount]

    def new_generation(self):
        # initialize the new generation of solutions by getting the best 20% of solutions from last gen
        new_gen = self.best_sols()
        for i in range(int(0.3 * POPULATION)):
            # add another 30% of randomly generated solutions to the new generation
            new_gen.append(gen_random_solution(self.size, self.matrix, self.indexes))
        while len(new_gen) < POPULATION:
            # create new solutions by performing cross-overs on two randomly selected solutions from the new gen
            new_sol = self.crossover_sol(get_random_element(new_gen), get_random_element(new_gen))
            # draw a random integer and mutate the current solution if it's lower than the probability for mutation
            mutate = randint(0, 100)
            new_sol = self.mutate_sol(new_sol) if mutate < MUTATION_PROB else new_sol
            new_gen.append(new_sol)
        return new_gen

    def solve(self):
        generation = 1
        while generation < 501:
            # calculate the fitness for all solutions
            self.sols_fit = self.fitness()

            # display the best solution in the current gen, with information about best, avg and worst fitness values.
            self.gui.show_solution(self.gen[self.index_best_sol], self.constraints, generation,
                                   self.best_fit, self.avg_fit, self.worst_fit)
            # create the next generation
            self.gen = self.new_generation()

            # we want to change mutation according to the generation to make spread "wave" effect, where in those
            # waves there will be spread higher possibility for mutations. that's why, for every 100 generations, the
            # first 20 will have spread higher change for mutation.
            if generation % 100 < 20:
                self.mutation_prob = 60
            else:
                self.mutation_prob = 10
            generation += 1
        self.gui.pause = True
        self.gui.gui_paused()


# A class inheriting GeneticAlg that has 2 functions used by both DarwinAlg and LemarkAlg to optimize their solutions.
class OptimizingAlg(GeneticAlg):
    def __init__(self, path):
        super().__init__(path)

    # preform optimization for single solution
    def single_sol_opt(self, sol):
        max_score = self.calc_score(sol)
        i = 0
        before_opt = self.calc_score(sol)
        for c in self.constraints:  # loop through all constraints
            if i == self.size:  # we allow optimization step as the size of the matrix
                break
            # if the constraint doesn't met, switch between the values in the cells
            if not is_constraint_maintained(c, sol):
                x1, y1 = c[0]  # left value, which should be bigger then right value, but currently he's smaller
                x2, y2 = c[1]  # right value, which should be smaller then left value, but currently he's bigger
                # optimize by switching between left value and right value in order to maintain the constraint
                temp = sol[x1 - 1][y1 - 1]
                sol[x1 - 1][y1 - 1] = sol[x2 - 1][y2 - 1]
                sol[x2 - 1][y2 - 1] = temp
                after_opt = self.calc_score(sol)  # calculate the score after the optimization step
                # if the new score is higher, assign it to the max_score variable and keep the optimize solution
                if after_opt > before_opt and after_opt > max_score:
                    max_score = after_opt
                # otherwise, revert the solution to it's original state
                elif before_opt > after_opt and before_opt > max_score:
                    max_score = before_opt
                    sol[x2 - 1][y2 - 1] = sol[x1 - 1][y1 - 1]
                    sol[x1 - 1][y1 - 1] = temp
                i += 1
        return sol, max_score

    # preform optimization for all solutions
    def optimize(self):
        self.sols_fit = []
        self.best_fit = float('-inf')
        self.worst_fit = float('inf')
        self.index_best_sol = -1
        opt_sols=[]
        sum_fitness = 0
        index = 0
        for sol in self.gen:
            # calculate the score for sol, and add it to sols_fit member
            opt_sol, sol_score = self.single_sol_opt(sol)  # optimize current solution
            opt_sols.append(opt_sol)
            self.sols_fit.append(sol_score)
            sum_fitness += sol_score
            # update the worst_fit and best_fit variables accordingly. if we update best_fit, we want to keep the index
            # for the best solution, so we also update that member
            if sol_score < self.worst_fit:
                self.worst_fit = sol_score
            if sol_score >= self.best_fit:
                self.best_fit = sol_score
                self.index_best_sol = index
            index += 1
        self.avg_fit = "%.2f" % (sum_fitness / POPULATION)
        return opt_sols


# Darwin Genetic Algorithm optimize all the solutions, calculate fitness of the optimized solutions,
# but the next generation will create based on the original solution, before optimization
class DarwinAlg(OptimizingAlg):
    def __init__(self, path):
        super().__init__(path)

    # calculate fitness of optimized solutions while maintaining that new generation creation will be based on original
    def fitness(self):
        temp = self.gen.copy()  # keep the original on temp variable for future use (new generation creation)
        self.gen = self.optimize()  # assign optimized solutions to self.gen, so we can use the calc_fitness method
        sols_fitness = self.calc_fitness()  # calculate the optimized solutions fitness
        self.gen = temp  # since the new_generation method uses self.gen to create new generation, assign original back
        return sols_fitness


# Lemark Genetic Algorithm optimize all the solutions, calculate fitness of the optimized solutions,
# and the next generation will create based on the optimized solutions
class LemarkAlg(OptimizingAlg):
    def __init__(self, path):
        super().__init__(path)

    # calculate fitness of optimized solutions
    def fitness(self):
        self.gen = self.optimize()  # assign optimized solutions to self.gen since the algorithm continues with them
        sols_fitness = self.calc_fitness()
        return sols_fitness


if __name__ == '__main__':

    assert len(sys.argv) == 3, "Incorrect call used when trying to run the program from the command line!"
    alg_type = sys.argv[2]
    if alg_type.lower() == "regular":
        ga = GeneticAlg(sys.argv[1])
    elif alg_type.lower() == "darwin":
        ga = DarwinAlg(sys.argv[1])
    elif alg_type.lower() == "lemark":
        ga = LemarkAlg(sys.argv[1])
    else:
        print("Wrong algorithm type entered. Exiting program.")
        sys.exit()

    ga.solve()
