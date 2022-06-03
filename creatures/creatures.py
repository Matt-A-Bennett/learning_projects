import numpy as np
import matplotlib.pyplot as plt

def evolve_plants(plant_list, ndays, lifespan=2, population_cap=200):
    for day in range(ndays):
        # make next generation
        nextgen_plant_list = []
        for plant in plant_list:
            # plants age
            plant.grow_up()
            # maybe dies of old age
            if plant.get_age() > lifespan:
                plant_list.remove(plant)
                continue
            # limit population
            if len(plant_list) > population_cap:
                continue
            # a new plant is born
            nextgen_plant_list.append(plant.reproduce())
        # add new generation to what's left of the previous generation
        plant_list += nextgen_plant_list
    return plant_list

# contents will be a np array, with a one for a plant, and a 2 for an animal
# an animal will be able to 'get_local_contents' when it can see
class World:
    def __init__(self, content):
        self.content = content
    def get_local_contents(self, position, radius):
        pass
    def update(self):
        pass

class Creature:
    def __init__(self, position=np.array([0,0]), energy=100, age=0):
        self.position = position
        self.energy = energy
        self.age = age
    def get_position(self):
        return self.position
    def get_energy(self):
        return self.energy
    def get_age(self):
        return self.age
    def grow_up(self):
        self.age += 1

class Plant(Creature):
    def __init__(self, position=np.array([0,0]), energy=100, age=0):
        super().__init__(position, energy, age)
    def reproduce(self):
        return Plant(self.position+np.random.randint(-10, 10, 2))

class Animal(Creature):
    def __init__(self, position=np.array([0,0]), energy=100, eyesight=5, speed=1):
        super().__init__(position, energy)
        self.eyesight = eyesight
        self.speed = speed
    def get_eyesight(self):
        return self.eyesight
    def get_speed(self):
        return self.speed
    def move(self, vector):
        self.position += vector
        self.energy -= sum(vector)
    def consume(self, amount):
        self.energy += amount
        if self.energy >= 100:
            self.energy = 100
    def visual_assess(self, world):
        pass # label value of things it can see
        # value*f(distance) (the distance is a metric I'll have them
        # calculate, but it will be up to them to mutate a useful f() to make
        # the value emphasised optimally
        # maybe each animal will carry a dictionary mapping each object type to
        # a value (which would be modified by it's distance)
        # {plant: mu * mu/distance * mu/self.energy,
        #  predator: mu * -mu/distance * mu/self.energy}

    def decision(self,assessment):
        pass #
        # for mu in mutations: # list of say 3 mutations would tell the animal
        # which values to consider (you could have one that looks at the 2 most
        # valuable and the lease valuable (i.e. dangerous) and from their
        # locations decides how to move
        # consider = [val for val in sort(assessment.get_values)[mu]]
        # then separate mutations would vote for moving the x,y toward or away
        # from the values considered...


plt.axis((-100, 100, -100, 100))
plt.grid(True)
flowers = [Plant()]
flowers = evolve_plants(flowers, ndays=2, lifespan=2, population_cap=200)
for flower in flowers:
    plt.plot(flower.get_position()[0],
             flower.get_position()[1], '.-g')

plt.show()

blob = Creature()
print(blob.get_energy())
print(blob.get_age())
print(blob.get_position())
print("----------")
animal = Animal()
print(animal.get_eyesight())
print(animal.get_speed())
print("----------")
hotdog = Animal(energy=70)
print(hotdog.get_position())
print(hotdog.get_eyesight())
print(hotdog.get_speed())
print(hotdog.get_energy())
hotdog.move(np.array([1,3]))
plt.plot(hotdog.get_position()[0], hotdog.get_position()[1], '.-')
print(hotdog.get_position())
print(hotdog.get_energy())
hotdog.consume(20)
print(hotdog.get_energy())
print(hotdog.get_age())
hotdog.grow_up()
print(hotdog.get_age())
hotdog.grow_up()
print(hotdog.get_age())
hotdog.consume(20)
print(hotdog.get_energy())
