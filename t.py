from Final_Version.Models import Generator

#This is the filepath to the model weights that were saved during training
FILEPATH = ""

#Turn the labels into classes
def words_to_classes(labels):
    return 1

#Somehow get the label using an api call
def get_label():

    return []


gen = Generator.build_generator()
gen.load_weights(FILEPATH)
labels = get_label()
class_to_be_generated = words_to_classes(labels)
generated_image = gen()
