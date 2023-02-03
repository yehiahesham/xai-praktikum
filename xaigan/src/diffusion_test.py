from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from models.generators import DiffusionModel

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from matplotlib import image as mpimg
    device = 'cuda'
    

    generator = DiffusionModel()
    #generator.to(device)
    
    text_inputs=[
        "the flowers and pedals of this flower look like a butterfly and they are spread out.",
        "A dog chewing on a mans shoes as he lays on a field.",
        "A female wearing a helmet is riding a horse down a street with stone buildings behind her.",
        "A glass of beer is sitting next to a vase full of flowers.",
    ]
    #generator.eval()
    for x in text_inputs:
        image = generator.forward(x)
        #print(image.shape)
        image.save("D:\\Praktikum\\xai_demo\\xai-praktikum\\xaigan\\src\\results\\diffusion" + "\\"+str(x)+'.png')