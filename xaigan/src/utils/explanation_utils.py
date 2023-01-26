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
           
            
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # For OpenMP error

            explainer = IntegratedGradients(discriminator)
            explanation = explainer.attribute(sample)

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
        exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
        lime = Lime(model,interpretable_model=SkLearnLinearRegression(),similarity_func=exp_eucl_distance)
        explanation = lime.attribute(sample,target=0,n_samples=200,perturbations_per_eval=100,show_progress=True)
    
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
