import torch
import os
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from captum.attr import DeepLiftShap,DeepLift, Saliency, IntegratedGradients, ShapleyValueSampling, Lime,LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from utils.vector_utils import values_target,images_to_vectors
# from lime import lime_image
# from skimage.segmentation import mark_boundaries


# defining global variables
global values
global discriminatorLime


def get_explanation(generated_data, discriminator, prediction, XAItype="shap", cuda=True, trained_data=None,
                    data_type="mnist") -> None:

    # initialize temp values to all 1s
    temp = values_target(size=generated_data.size(), value=1.0, cuda=cuda)

    # mask values with low prediction
    mask = (prediction < 0.5).view(-1)
    indices = (mask.nonzero(as_tuple=False)).detach().cpu().numpy().flatten().tolist()

    data = generated_data[mask, :]

    if len(indices) > 1:
        if XAItype == "saliency":
            for i in range(len(indices)):
                explainer = Saliency(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0))

        elif XAItype == "shap":
            for i in range(len(indices)):
                explainer = DeepLiftShap(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0), trained_data, target=0)

        elif XAItype == "integrated_gradients":
            pass

        # elif XAItype == "lime":
        #     explainer = lime_image.LimeImageExplainer()
        #     global discriminatorLime
        #     discriminatorLime = deepcopy(discriminator)
        #     discriminatorLime.cpu()
        #     discriminatorLime.eval()
        #     for i in range(len(indices)):
        #         if data_type == "cifar":
        #             tmp = data[i, :].detach().cpu().numpy()
        #             tmp = np.reshape(tmp, (32, 32, 3)).astype(np.double)
        #             exp = explainer.explain_instance(tmp, batch_predict_cifar, num_samples=100)
        #         else:
        #             tmp = data[i, :].squeeze().detach().cpu().numpy().astype(np.double)
        #             exp = explainer.explain_instance(tmp, batch_predict, num_samples=100)
        #         _, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, negative_only=False)
        #         temp[indices[i], :] = torch.tensor(mask.astype(np.float))
        #     del discriminatorLime
        else:
            raise Exception("wrong xAI type given")

    if cuda:
        temp = temp.cuda()
    set_values(normalize_vector(temp))


def extract_explanation(model,sample,type):
    if type == "saliency":
                explainer = Saliency(model)
                explanation = explainer.attribute(sample)

    elif type == "shapley_value_sampling":
        explainer = ShapleyValueSampling(model)
        explanation = explainer.attribute(sample, n_samples=2)
        
    elif type == "integrated_gradients":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # For OpenMP error
        explainer = IntegratedGradients(model)
        explanation = explainer.attribute(sample)

    elif type == "deeplift":
        # Delete inplace=True from ReLU's in model to work, otherwise crashes
        explainer = DeepLift(model)
        explanation = explainer.attribute(sample)
        
    elif type == "lime":
        feature_mask,kernel_sz,id =None,1, 0
        if kernel_sz>1:
            w,h=sample.shape[2],sample.shape[3]  #ex:2 , then 2x2 feature mask  
            e_h = w - kernel_sz +1 
            e_w = h - kernel_sz +1
            feature_mask=torch.zeros(w,h,dtype=torch.long) # mask like our image , ex: 32x32
            for i in range(0,e_h,kernel_sz): #32x32 image
                for j in range(0,e_w,kernel_sz):
                    feature_mask[i][j]=id; feature_mask[i+1][j]=id
                    feature_mask[i][j+1]=id; feature_mask[i+1][j+1]=id
                    id+=1
        
        exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
        lime = Lime(model,interpretable_model=SkLearnLinearRegression(),similarity_func=exp_eucl_distance)
        explanation = lime.attribute(sample,target=0,feature_mask=feature_mask,n_samples=50,perturbations_per_eval=100,show_progress=True)
    
    elif type == 'lime2':
        global discriminatorLime
        discriminatorLime = deepcopy(model)
        discriminatorLime.cpu()
        discriminatorLime.eval()
        samplelime = sample.permute(0,2,3,1).detach().numpy().astype(np.double).squeeze()
        
        def predict(images):
            images = np.transpose(images, (0, 3, 1, 2)) # stack up all images
            batch = torch.stack([i for i in torch.Tensor(images)], dim=0)
            prob = discriminatorLime(batch)
            return prob.view(-1).unsqueeze(1).detach().numpy()
        explainer = lime_image.LimeImageExplainer()

        exp = explainer.explain_instance(image=samplelime, classifier_fn = predict, labels = (1), num_samples=100)

        temp, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, num_features=5, hide_rest=False,min_weight = 0.0)
        #temp4, mask4 = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, num_features=1, hide_rest=False,min_weight = 0.0)
        #temp2, mask2 = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=True,min_weight = 0.0)
        #temp3, mask3 = exp.get_image_and_mask(exp.top_labels[0], positive_only =False,negative_only =True, num_features=5, hide_rest=True,min_weight = 0.0)
        #plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        img_boundry = mark_boundaries(temp/2 + 0.5, mask).astype(float)
        #img_boundry2 = mark_boundaries(temp2/2 + 0.5, mask2).astype(float)
        #img_boundry3 = mark_boundaries(temp3/2 + 0.5, mask3).astype(float)
        #img_boundry4 = mark_boundaries(temp4/2 + 0.5, mask4).astype(float)
        explanation = torch.from_numpy(img_boundry).permute(2,0,1)
            
    return explanation



def explanation_hook(module, grad_input, grad_output):
    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """
    # get stored mask
    temp = get_values()
    # multiply with mask
    new_grad = grad_input[0] + 0.2 * (grad_input[0] * temp)

    return (new_grad, )


def normalize_vector(vector: torch.tensor) -> torch.tensor:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector -= vector.min()
    vector /= vector.max()
    vector[torch.isnan(vector)] = 0
    return vector.type(torch.float32)


def get_values() -> np.array:
    """ get global values """
    global values
    return values


def set_values(x: np.array) -> None:
    """ set global values """
    global values
    values = x
