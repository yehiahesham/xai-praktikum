from get_data import get_loader
from utils.vector_utils import  values_target, weights_init, vectors_to_images_coco,vectors_to_images, noise_coco
from evaluation.evaluate_generator_coco import calculate_metrics_coco, read_random_captionsFile,get_random_text
from logger import Logger
from utils.explanation_utils import get_explanation,explanation_hook
from torch.autograd import Variable
from torch import nn
import torch
import time
import random
import string
import pandas as pd
import numpy as np




class Experiment:
    """ The class that contains the experiment details """
    def __init__(self, experimentType):
        """
        Standard init
        :param experimentType: experiment enum that contains all the data needed to run the experiment
        :type experimentType:
        """

        #Extract Paramters from Experiment Enum
        self.name = experimentType.name
        self.type = experimentType.value

        self.explainable = self.type["explainable"]
        self.explanationType = self.type["explanationType"]
        self.noise_emb_sz = self.type["noise_emb_sz"]
        self.text_max_len = self.type["text_max_len"]
        self.Encoder_emb_sz = self.type["Encoder_emb_sz"] #Hyper-paramter (encoder output/ generator input)
        self.text_emb_sz = self.type["text_emb_sz"] #TODO: to be able to chanhe that in roberta
        self.target_image_w = self.type["target_image_w"] #TODO: to be able to chanhe that in roberta
        self.target_image_h = self.type["target_image_h"] #TODO: to be able to chanhe that in roberta
        
        self.use_CLS_emb = self.type["use_CLS_emb"]
        self.use_captions = self.type["use_captions"]  #True: use noise and captions, False: only use noise 
        self.use_captions_only = self.type["use_captions_only"] #True: only use captions wihtou noise, #False: use noise and captions (need use_captions=True)
        self.use_one_caption = self.type["use_one_caption"] #with noise and captions, True: use 1 caption/image,  False: use Mult. captions/image. (need use_captions=True)
        
        
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Flag for embedding network
        self.text_emb_model = None

        # Declare & intialize Models
        if self.use_captions:
            self.text_emb_model = self.type["text_emb_model"]().to(self.device)
            self.text_emb_model.eval()
            #freezing text encoder weights
            for param in self.text_emb_model.parameters():
                param.requires_grad = False

            if  self.use_captions_only==False: #noise+Captions    
                self.generator = self.type["generator"](
                    noise_emb_sz=self.noise_emb_sz,
                    text_emb_sz=self.text_emb_sz,
                    n_features=self.Encoder_emb_sz).to(self.device) 
            
            else: #self.use_captions_only: #Captions 
                self.generator = self.type["generator"](n_features=self.text_emb_sz).to(self.device)
                                
        else: #noise only
            self.generator = self.type["generator"](n_features=self.noise_emb_sz).to(self.device)

        self.discriminator = self.type["discriminator"]().to(self.device)


        self.g_optim = self.type["g_optim"](self.generator.parameters(), lr=self.type["glr"], betas=(0.5, 0.99))
        self.d_optim = self.type["d_optim"](self.discriminator.parameters(), lr=self.type["dlr"], betas=(0.5, 0.99))
        self.loss = self.type["loss"]
        self.epochs = self.type["epochs"]
        self.cuda = True if torch.cuda.is_available() else False
        self.real_label = 0.9
        self.fake_label = 0.1
        self.samples = 16
        torch.backends.cudnn.benchmark = True

    def run(self, logging_frequency=4):
        """
        This function runs the experiment
        :param logging_frequency: how frequently to log each epoch (default 4)
        :type logging_frequency: int
        :return: None
        :rtype: None
        """
        
        start_time = time.time()

        explanationSwitch = (self.epochs + 1) / 2 if self.epochs % 2 == 1 else self.epochs / 2

        logger = Logger(self.name, self.type["dataset"])


        sampling_args = {
            'generator' :self.type["generator"],
            'pretrained_generatorPath': f'{logger.data_subdir}/generator.pt',
            "pretrained_discriminatorPath": f'{logger.data_subdir}/discriminator.pt',
            "text_emb_model"  :self.type["text_emb_model"],           
            'dataset'       :self.type["dataset"],
            'noise_emb_sz'  :self.type["noise_emb_sz"],
            'text_emb_sz'   :self.type["text_emb_sz"],
            'Encoder_emb_sz':self.Encoder_emb_sz,
            'discriminator': self.type["discriminator"],

            'use_captions'  :self.type['use_captions'],
            'use_one_caption'  :self.type['use_one_caption'],
            'use_captions_only'  :self.type['use_captions_only'],
            
            "explainable" :self.type["explainable"],
            "explanationType" : self.type["explanationType"],
            "explanationTypes" : ["lime","integrated_gradients", "saliency", "shapley_value_sampling","deeplift"],
            
        }

        # calculate_metrics_coco(sampling_args,numberOfSamples=15)
        # return

        test_noise = noise_coco(self.samples, self.cuda)
        if self.use_captions: #will need captions, extract text emb
            if self.use_one_caption: #1-captions/image
                random_captions = read_random_captionsFile(self.type["dataset"])
                random_texts = get_random_text(self.samples,random_captions, self.type["dataset"]) 
                test_texts_embs = torch.stack([self.text_emb_model.forward([random_texts[i]]).squeeze() \
                                                for i in range(self.samples)], dim=0)
            else : # Mult-captions/image
                pass 
        
        else:   #noise only
            test_dense_emb=test_noise
        
        
        
        if  self.use_captions and self.use_captions_only: #captions only
            test_dense_emb=test_texts_embs[:,:,np.newaxis, np.newaxis]

        elif self.use_captions and self.use_captions_only==False: #noise + captions
            #concatinate the 2 embeddings if needed
            test_dense_emb = torch.cat( (test_texts_embs,test_noise.reshape(self.samples,-1)), 1)

        
        

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        loader = get_loader(self.type["batchSize"], self.type["percentage"], self.type["dataset"], self.target_image_w, self.target_image_h)
        num_batches = len(loader)

        if self.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.loss = self.loss.cuda()

        if self.explainable:
            trained_data = Variable(next(iter(loader))[0])
            if self.cuda:
                trained_data = trained_data.cuda()
        else:
            trained_data = None

        # track losses
        G_losses = []
        D_losses = []
        best_g_error=10000
        best_d_error=10000

        local_explainable = False

        # Start training
        for epoch in range(1, self.epochs + 1):

            if self.explainable and (epoch - 1) == explanationSwitch:    
                self.generator.out.register_backward_hook(explanation_hook)
                local_explainable = True            
            
            for n_batch, (real_batch) in enumerate(loader):
                
                if  self.type["dataset"]=='mscoco' :
                    N = len(real_batch)
                    batch_images = torch.stack([real_batch[i][0] for i  in range(0,N)] , dim=0)
                    batch_images = batch_images.reshape((N, 3, self.target_image_w, self.target_image_h))

                    # lables       = torch.stack([real_batch[i][1] for i  in range(0,N)] , dim=0)

                    # batch_images= real_batch[0][0] #1st - (image)
                    # lables= real_batch[0][1]       #1st - (5 captions/image)
                    
                    
                    # for i in range(1,N): #looping on images, and aggregate the tensor array
                    #     batch_images = torch.stack( [batch_images,real_batch[i][0]], dim=0)
                    
                    # for i in range(1,N): #looping on capions, and aggregate the tensor array
                    #     lables = torch.cat( (batch_images,real_batch[i][1]), 0)

                elif self.type["dataset"] == 'flowers-102':
                    batch_images,labels,captions = real_batch #images,classes, Matrix of (10,batch_size) =>  10 cap/per image
                    N = batch_images.size(0)
                    MAX_CAPTIONS_PER_IMAGE=10

                    #text proccess if needed 
                    texts_embs = None
                    if   self.use_captions and self.use_one_caption:
                        batched_captions = []
                        # pick best caption as captions with longest length
                        for batch_elt_i in range(N):   #loop on batch elements
                            caption_with_max_len=max([captions[caption_j][batch_elt_i] for caption_j in range (0,MAX_CAPTIONS_PER_IMAGE)] ,key=len)
                            batched_captions.append(caption_with_max_len)                            
                    
                        #Stack text embs [batch_size,text_emb_sz]
                        texts_embs = torch.stack([self.text_emb_model.forward([batched_captions[i]]).squeeze() \
                                            for i in range(N)], dim=0)
                  
                    elif self.use_captions and not self.use_one_caption:
                        #TODO: Need to be tested
                        for batch_elt_i in N:   #loop on batch elements
                            singleImage_Multi_captions = [captions[caption_j][batch_elt_i] for caption_j in range (0,MAX_CAPTIONS_PER_IMAGE)]
                            singleImage_Multi_Captions_Emb = torch.stack([self.text_emb_model.forward(singleImage_Multi_captions).squeeze() \
                                            for i in range(N)], dim=0)
                            
                            #Aggregation function: averging (Hyperparamerte\r) 
                            singleImage_Multi_Captions_Emb = singleImage_Multi_Captions_Emb.mean()

                            #build the batched result 
                            texts_embs = torch.stack([texts_embs,singleImage_Multi_Captions_Emb], dim=0)
                    

                else : #other datasets
                    batch_images, labels = real_batch
                    N = batch_images.size(0)
                
                # 0. Pass (Text+Noise) embeddings >  EmbeddingEncoder_NN > Generator_NN

                noise_emb = noise_coco(N, self.cuda)
                if self.use_captions and self.use_captions_only==False: #noise + captions
                    # concatinate the 2 embeddings
                    dense_emb = torch.cat((texts_embs,noise_emb.reshape(N,-1)), 1)#.to(torch.float16)
                
                elif self.use_captions and self.use_captions_only: #captions only
                    dense_emb=texts_embs[:,:,np.newaxis, np.newaxis]
                    
                else: #noise only
                    dense_emb = noise_emb

                # 1. Train Discriminator
                # Generate fake data and detach (so gradients are not calculated for generator)
                fake_data = self.generator(dense_emb).detach()                

                if self.cuda:
                    batch_images = batch_images.cuda()
                    fake_data = fake_data.cuda()

                # Train D
                
                d_error, d_pred_real, d_pred_fake = self._train_discriminator(real_data=batch_images, fake_data=fake_data)

		        
                # 2. Train Generator
                # Generate fake data
                
                noise_emb = noise_coco(N, self.cuda) #new noise emb
                
                if self.use_captions and self.use_captions_only==False: #noise + captions
                    #concatinate the SAME text embeddings
                    dense_emb = torch.cat((texts_embs,noise_emb.reshape(N, -1)), 1)
                
                elif self.use_captions and self.use_captions_only: #captions only
                    dense_emb=texts_embs[:,:,np.newaxis, np.newaxis]
                
                else: #noise only
                    dense_emb=noise_emb
               
                fake_data = self.generator(dense_emb) #generate a new fake image to train the Generator & Encoder

                if self.cuda:
                    fake_data = fake_data.cuda()

                # Train G (with Encoder Network if exist)
                g_error = self._train_generator(fake_data=fake_data, local_explainable=local_explainable,
                                                trained_data=trained_data)
                
                            
                # Save models if their losses are smaller 
                if(g_error<=best_g_error):
                    logger.save_model(model=self.generator,name="generator",epoch=epoch,loss=g_error)
                    best_g_error=g_error
                if(d_error<=best_d_error):
                    logger.save_model(model=self.discriminator,name="discriminator",epoch=epoch,loss=d_error)
                    best_d_error=d_error
                
               
                
                # Save Losses for plotting later
                G_losses.append(g_error.item())
                D_losses.append(d_error.item())

                logger.log(d_error, g_error, epoch, n_batch, num_batches)

                # Display status Logs
                if n_batch % (num_batches // logging_frequency) == 0:
                    logger.display_status(
                        epoch, self.epochs, n_batch, num_batches,
                        d_error, g_error, d_pred_real, d_pred_fake
                    )
            # logger.Generator_per_epoch(fake_data[0], epoch)

        logger.save_errors(g_loss=G_losses, d_loss=D_losses)
        timeTaken = time.time() - start_time
        test_images = self.generator(test_dense_emb)

        
        
        
        test_images = vectors_to_images(test_images,self.target_image_w,self.target_image_h).cpu().data    
        calculate_metrics_coco(sampling_args,numberOfSamples=15)
        
        logger.log_images(test_images, self.epochs + 1, 0, num_batches)
        logger.save_scores(timeTaken, 0)
        return

    def _train_generator(self, fake_data: torch.Tensor, local_explainable, trained_data=None) -> torch.Tensor:
        """
        This function performs one iteration of training the generator
        :param fake_data: tensor data created by generator
        :return: error of generator on this training step
        """
        N = fake_data.size(0)

        # Reset gradients
        self.g_optim.zero_grad()

        # Sample noise and generate fake data
        prediction = self.discriminator(fake_data).view(-1)

        if local_explainable:
            get_explanation(generated_data=fake_data, discriminator=self.discriminator, prediction=prediction,
                            XAItype=self.explanationType, cuda=self.cuda, trained_data=trained_data,
                            data_type=self.type["dataset"])

        # Calculate error and back-propagate
        error = self.loss(prediction, values_target(size=(N,), value=self.real_label, cuda=self.cuda))

        error.backward()

        # clip gradients to avoid exploding gradient problem
        nn.utils.clip_grad_norm_(self.generator.parameters(), 10)

        # update parameters
        self.g_optim.step()

        # Return error
        return error

    def _train_discriminator(self, real_data: Variable, fake_data: torch.Tensor):
        """
        This function performs one iteration of training the discriminator
        :param real_data: batch from dataset
        :type real_data: torch.Tensor
        :param fake_data: data from generator
        :type fake_data: torch.Tensor
        :return: tuple of (mean error, predictions on real data, predictions on generated data)
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        N = real_data.size(0)
        real_data = real_data.float()#.to(torch.float16)

        # Reset gradients
        self.d_optim.zero_grad()

        # 1.1 Train on Real Data
        prediction_real = self.discriminator(real_data).view(-1)#.to(torch.half)

        # Calculate error
        error_real = self.loss(prediction_real, values_target(size=(N,), value=self.real_label, cuda=self.cuda))

        # 1.2 Train on Fake Data
        prediction_fake = self.discriminator(fake_data).view(-1)#.to(torch.float16)

        # Calculate error
        error_fake = self.loss(prediction_fake, values_target(size=(N,), value=self.fake_label, cuda=self.cuda))

        # Sum up error and backpropagate
        error = error_real + error_fake
        error.backward()

        # 1.3 Update weights with gradients
        self.d_optim.step()

        # Return error and predictions for real and fake inputs
        return (error_real + error_fake) / 2, prediction_real, prediction_fake
