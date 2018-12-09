function normalize_data


 inputData = dlmread('my_peds_annotations_interpolated_raw.txt'); 
 
 frameWidth  =  720; 
 frameHeight = 576 ; 
 
 %normalize (x,y) locations into range [0,1]
 normalizedData = inputData ; 
 normalizedData(:,3) = normalizedData(:,3)/frameWidth ; % normalize x
 normalizedData(:,4) = normalizedData(:,4)/frameHeight ; % normalize y

csvwrite('my_peds_annotations_normalize_[0,1].txt', normalizedData);

end 