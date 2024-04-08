import numpy as np
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN

# SOURCE: Demetrius Gulewicz
def jpg_to_face_train(input_dir, i_start, num_img):
    bad_id = []
    valid_id = []
    valid_bbox = []
    valid_q = []
    valid_ar = []
    valid_px = []
    
    # Iterate over RGB images in input directory   
    mtcnn = MTCNN(margin=40,select_largest=False, keep_all=True, post_process=False)
        
    for i in range(num_img):
        if i % 1000 == 0:
            print(i)
            
        # get image
        img = Image.open(input_dir + str(i_start + i) + '.jpg')
        
        # convert image to RGB
        if img.mode == "P":
            img = img.convert("RGBA").convert("RGB")
        else:
            img = img.convert("RGB")
        
        # get all possible faces
        face = mtcnn.detect(img, landmarks=False)
            
        # extract only images that have exactly one face
        if (face[0] is not None) and (len(face[0]) == 1):
            # get bounding box
            bbox = np.array([max(round(face[0][0][0]),0), max(round(face[0][0][1]),0), min(round(face[0][0][2]),img.size[0]), min(round(face[0][0][3]),img.size[1])])
            
            # get aspect ratio
            ar = abs((bbox[0] - bbox[2]) / (bbox[1] - bbox[3]))
            
            # get number of pixels
            px = abs((bbox[0] - bbox[2]) * (bbox[1] - bbox[3]))
            
            # save statistics
            valid_id.append(i_start + i)
            valid_q.append(face[1][0])
            valid_bbox.append(bbox)
            valid_ar.append(ar)
            valid_px.append(px)
        else:
            bad_id.append(i_start + i)
            
        # close image
        img.close()
        del img

    return valid_id, valid_bbox, valid_q, valid_ar, valid_px, bad_id

# SOURCE: Demetrius Gulewicz
def jpg_to_face_test(input_dir, i_start, num_img):
    # define lists
    valid_bbox = []
    valid_q = []
    valid_ar = []
    valid_px = []
    dup_number = []
    
    # initialize dup index
    dup_idx = 0
    
    # Iterate over RGB images in input directory   
    mtcnn = MTCNN(margin=40,select_largest=False, keep_all=True, post_process=False)
        
    for i in range(num_img):
        if i % 1000 == 0:
            print(i)
            
        # get image
        img = Image.open(input_dir + str(i_start + i) + '.jpg')
        
        # convert image to RGB
        if img.mode == "P":
            img = img.convert("RGBA").convert("RGB")
        else:
            img = img.convert("RGB")
        
        # get all possible faces
        face = mtcnn.detect(img, landmarks=False)
            
        # if no face is detected, then take the image as is
        if (face[0] is None):
            # get bounding box
            bbox = np.array([0, 0, img.size[0], img.size[1]])
             
            # get aspect ratio
            ar = abs((bbox[0] - bbox[2]) / (bbox[1] - bbox[3]))
            
            # get number of pixels
            px = abs((bbox[0] - bbox[2]) * (bbox[1] - bbox[3]))
            
            # save statistics
            valid_q.append(-1)
            valid_bbox.append(bbox)
            valid_ar.append(ar)
            valid_px.append(px)
            dup_number.append(dup_idx)
            
            # update dup number
            dup_idx = dup_idx + 1
             
        # extract images that have exactly one face
        elif (len(face[0]) == 1):
            # get bounding box
            bbox = np.array([max(round(face[0][0][0]),0), max(round(face[0][0][1]),0), min(round(face[0][0][2]),img.size[0]), min(round(face[0][0][3]),img.size[1])])
            
            # get aspect ratio
            ar = abs((bbox[0] - bbox[2]) / (bbox[1] - bbox[3]))
            
            # get number of pixels
            px = abs((bbox[0] - bbox[2]) * (bbox[1] - bbox[3]))
            
            # save statistics
            valid_q.append(face[1][0])
            valid_bbox.append(bbox)
            valid_ar.append(ar)
            valid_px.append(px)
            dup_number.append(dup_idx)
            
            # update dup number
            dup_idx = dup_idx + 1
            
        # if there are multiple faces, extract all faces
        else:
            num_faces = len(face[0])
            bbox = []
            ar = []
            px = []
            
            for i in range(num_faces):
                # get bounding box
                bbox = np.array([max(round(face[0][i][0]),0), max(round(face[0][i][1]),0), min(round(face[0][i][2]),img.size[0]), min(round(face[0][i][3]),img.size[1])])
                
                # get aspect ratio
                ar = abs((bbox[0] - bbox[2]) / (bbox[1] - bbox[3]))
                
                # get number of pixels
                px = abs((bbox[0] - bbox[2]) * (bbox[1] - bbox[3]))
            
                # save statistics
                valid_q.append(face[1][i])
                valid_bbox.append(bbox)
                valid_ar.append(ar)
                valid_px.append(px)
                dup_number.append(dup_idx)
            
            # update dup number
            dup_idx = dup_idx + 1
            
        # close image
        img.close()
        del img

    return dup_number, valid_bbox, valid_q, valid_ar, valid_px

# SOURCE: Demetrius Gulewicz
def filter_faces_train(input_dir, output_dir, valid_id, bbox, q, ar, px, j_max, ar_mx, ar_mn, q_mn, px_mn, npx):
    valid_train_filt = []
    
    for j in range(j_max):
        i = int(valid_id[j])
        if i % 1000 == 0:
            print(i)
            
        if (ar[j] < ar_mx) and (ar[j] > ar_mn) and (q[j] > q_mn) and (px[j] > px_mn):
            # get image
            img = Image.open(input_dir + str(i) + '.jpg')
            
            # convert image to RGB
            if img.mode == "P":
                img = img.convert("RGBA").convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            # crop image
            img_face = img.crop(bbox[j,:])
            
            # pad and scale image
            img_face = ImageOps.pad(img_face,(npx,npx))
            
            # save final image
            img_face.save(output_dir + str(i) + '.jpg')
            
            # construct data array
            data_row = np.hstack((i,q[j],ar[j],px[j],bbox[j,:]))
            
            # save data
            valid_train_filt.append(data_row)
    
    return valid_train_filt

# SOURCE: Demetrius Gulewicz
def filter_faces_test(input_dir, output_dir, dup_number, bbox, q, q_mn, npx):
    # all base image indexes
    j_max = int(max(dup_number)) + 1
    
    # initialize filtered duplicates
    dup_number_filt = []
    q_filt = []
    bbox_filt = []
    abs_idx = 0
    
    for j in range(j_max):
        idxs = np.where(dup_number == j)[0]
        num_idxs = len(idxs)
        
        if j % 1000 == 0:
            print(j)
            
        # get image
        img = Image.open(input_dir + str(j) + '.jpg')
        
        # convert image to RGB
        if img.mode == "P":
            img = img.convert("RGBA").convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
            
        # if only one possible choice, pick that choice
        if num_idxs == 1:
            # crop image
            img_face = img.crop(bbox[idxs[0],:])
            
            # pad and scale image
            img_face = ImageOps.pad(img_face,(npx,npx))

            # save final image
            img_face.save(output_dir + str(j) + '_0.jpg')
            q_filt.append(q[j])
            bbox_filt.append(bbox[idxs[0],:])
            dup_number_filt.append(j)
            abs_idx = abs_idx + 1
            
        # if multiple choices, pick only highly likely faces, if such faces exist
        else:
            # determine if filtering is ok
            if len(np.where(q[idxs] > q_mn)[0]) > 0:
                q_filter = True
            else:
                q_filter = False
            
            # run through all images, filtering where permissible
            work_idx = 0
            for i in range(num_idxs):
                if (q_filter) and (q[idxs[i]] > q_mn):
                    # crop image
                    img_face = img.crop(bbox[idxs[i],:])
                    
                    # pad and scale image
                    img_face = ImageOps.pad(img_face,(npx,npx))
                
                    # save final image
                    img_face.save(output_dir + str(j) + '_' + str(work_idx) + '.jpg')
                    q_filt.append(q[j])
                    bbox_filt.append(bbox[idxs[i],:])
                    dup_number_filt.append(j)
                    work_idx = work_idx + 1
                    abs_idx = abs_idx + 1
                    
                elif (not q_filter):
                    # crop image
                    img_face = img.crop(bbox[idxs[0],:])
                    
                    # pad and scale image
                    img_face = ImageOps.pad(img_face,(npx,npx))
                
                    # save final image
                    img_face.save(output_dir + str(j) + '_' + str(work_idx) + '.jpg')
                    q_filt.append(q[j])
                    bbox_filt.append(bbox[idxs[i],:])
                    dup_number_filt.append(j)
                    work_idx = work_idx + 1
                    abs_idx = abs_idx + 1
                else:
                    abs_idx = abs_idx + 1
                    
    return dup_number_filt, q_filt, bbox_filt

# SOURCE: Demetrius Gulewicz
def compute_mu_sig(train_dir, face_filt_dir):
    # initialize
    sum_mu = np.zeros((3,1))
    mu = np.zeros((3,1))
    sum_sig = np.zeros((3,1))
    sig = np.zeros((3,1))
    
    # get valid ids
    valid_train = np.loadtxt(face_filt_dir, delimiter =',')
    valid_id = valid_train[:,0]
    num_img = len(valid_id)
    
    # compute mean
    for i in range(num_img):
        img = Image.open(train_dir + str(int(valid_id[i])) + '.jpg')
        sum_i = (np.sum(np.array(img)/255, axis=(0,1)).reshape((3,1)))
        sum_mu = sum_mu + sum_i
        
    mu = sum_mu/(num_img*50176)
        
    # compute sigma
    for i in range(num_img):
        img = Image.open(train_dir + str(int(valid_id[i])) + '.jpg')
        img_arr = np.array(img)/255
        
        sig_i_R = np.sum(np.power(img_arr[:,:,0] - mu[0],2),axis=(0,1))/50176
        sig_i_G = np.sum(np.power(img_arr[:,:,1] - mu[1],2),axis=(0,1))/50176
        sig_i_B = np.sum(np.power(img_arr[:,:,2] - mu[2],2),axis=(0,1))/50176
        
        sum_sig_i = np.array([sig_i_R,sig_i_G,sig_i_B]).reshape((3,1))
        sum_sig = sum_sig + sum_sig_i
        
    sig = sum_sig/(num_img*50176)
        
    return mu, sig