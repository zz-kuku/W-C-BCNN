1.Setting according to the README_BCNN.md.

2.Modify some codes, except something introduced in the README_BCNN.md:
    When train:
    1. imdb_bcnn_train_dag.m
        line 37 to 51, setting whether using clickconnection layer and weakly 
        supervise weight for loss.
    2. initializeNetworkTwoStreams.m
        line 158 to 198, the network construction.
    3. matconvnet/matlab/vl_nnloss.m
        line 226 to 231, save the loss each batch.

    When test:
    4. model_train.m
        line 10, opts.isClickW is true if using the weakly supervised weight.
    5. get_bcnn_features.m
        line 96, select feat according the last layer of network.(except dropout)
    6. initializeNetFromEncoder.m
        line 27, the same as step 5.
    7. code*.m
        help to tune the parament.
    8. code_result.m
        help get the result of step 7.


3.Augment the dataset by using Do_Augmentation.m



                                                                                To be improved.

