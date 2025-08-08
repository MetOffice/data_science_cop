import ray
import time

database = [
    "Learning", "Ray",
    "Flexible", "Distributed", "Python", "for", "Machine", "Learning"
]


def retrieve(item):
    time.sleep(item / 10.)
    return item, database[item]



@ray.remote
def retrieve_task(item):
    return retrieve(item)

def print_runtime(input_data, start_time):
    print(f'Runtime: {time.time() - start_time:.2f} seconds, data:')
    print(*input_data, sep="\n")


start = time.time()
data = [retrieve(item) for item in range(8)]
print_runtime(data, start)

start = time.time()
object_references = [
    retrieve_task.remote(item) for item in range(8)
]
data = ray.get(object_references)
print_runtime(data, start)