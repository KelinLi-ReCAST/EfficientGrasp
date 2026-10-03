# B3 hyper-parameter grep over PSSN scripts

Command (run from `contact_point_selection/UniGrasp/point_set_selection/`):

```
for f in train_pssn.py point_set_selection.py unigrasp_train.py point_set_selection_test.py point_set_selection_test_with_gt.py point_set_selection_raw_point_cloud.py unigrasp.py; do echo "=== $f"; grep -n "^nnn\|^TOP_K\|TOP_K2 =\|batch_size',\|num_epochs'\|learning_rate=\|epoch == \|k=1024\|k=512\|\[None,256\|gripper_index = \|if __name__\|restore_stage3" $f | grep -v '^[0-9]*:\s*#'; done
```

```
=== train_pssn.py
51:nnn = 1
56:parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
57:parser.add_argument('--batch_size',type=int,default=nnn * 1, help='Number of examples within a batch')
71:TOP_K = 1024 
73:TOP_K2 = 1024
82:gripper_feat_tf = tf.placeholder(tf.float32,[None,256*3])
101:  train_op_nor = tf.train.AdamOptimizer(learning_rate=2 * 1e-3).minimize(loss_nor)
129:  train_op_stage1 = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss_stage1)
198:  train_op_stage2  = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage2)
410:  train_op_stage3  = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_stage3)
414:  out_corr_top_value_stage3_tf, out_corr_top_index_stage3_tf = tf.nn.top_k(out_corr_score_stage3_tf, k=1024, sorted=True) 
442:def restore_stage2_v2(epoch):
452:def restore_stage3_v1(epoch):
461:def restore_stage3_v2(epoch):
473:def restore_stage3(epoch):
485:  if epoch == 0:
549:      gripper_index = np.random.choice(np.array([11,12,13]),gripper_size,replace=False) 
1073:    if epoch == 220:    
1107:if __name__ == "__main__":
1110:  restore_stage3_v2(221)
=== point_set_selection.py
49:nnn = 3
54:parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
55:parser.add_argument('--batch_size',type=int,default=nnn * 1, help='Number of examples within a batch')
69:TOP_K = 1024 
71:TOP_K2 = 1024
80:gripper_feat_tf = tf.placeholder(tf.float32,[None,256 * 3])
99:  train_op_nor = tf.train.AdamOptimizer(learning_rate=2 * 1e-3).minimize(loss_nor)
127:  train_op_stage1 = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss_stage1)
196:  train_op_stage2  = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss_stage2)
408:  train_op_stage3  = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_stage3)
412:  out_corr_top_value_stage3_tf, out_corr_top_index_stage3_tf = tf.nn.top_k(out_corr_score_stage3_tf, k=1024, sorted=True) 
441:def restore_stage2_v2(epoch):
451:def restore_stage3_v1(epoch):
460:def restore_stage3_v2(epoch):
472:def restore_stage3(epoch):
484:  if epoch == 0:
540:      gripper_index = np.random.choice(np.array([11,12,13]),gripper_size,replace=False) 
1235:if __name__ == "__main__":
=== unigrasp_train.py
57:nnn = 3
62:parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
63:parser.add_argument('--batch_size',type=int,default=nnn * 1, help='Number of examples within a batch')
77:TOP_K = 1024 
79:TOP_K2 = 1024
88:gripper_feat_tf = tf.placeholder(tf.float32,[None,256 * 3])
107:  train_op_nor = tf.train.AdamOptimizer(learning_rate=2 * 1e-3).minimize(loss_nor)
135:  train_op_stage1 = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss_stage1)
204:  train_op_stage2  = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss_stage2)
416:  train_op_stage3  = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_stage3)
420:  out_corr_top_value_stage3_tf, out_corr_top_index_stage3_tf = tf.nn.top_k(out_corr_score_stage3_tf, k=1024, sorted=True) 
449:def restore_stage2_v2(epoch):
459:def restore_stage3_v1(epoch):
468:def restore_stage3_v2(epoch):
480:def restore_stage3(epoch):
492:  if epoch == 0:
548:      gripper_index = np.random.choice(np.array([12,13,11]),gripper_size,replace=False) 
1242:if __name__ == "__main__":
=== point_set_selection_test.py
45:nnn = 1
51:parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
52:parser.add_argument('--batch_size',type=int,default=nnn * 1,help='Number of examples within a batch')
67:TOP_K = 1024
68:TOP_K2 = 1024
76:gripper_feat_tf = tf.placeholder(tf.float32,[None,256*3])
95:  train_op_nor = tf.train.AdamOptimizer(learning_rate=2 * 1e-3).minimize(loss_nor)
122:  train_op_stage1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage1)
191:  train_op_stage2  = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage2)
403:  train_op_stage3  = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_stage3)
407:  out_corr_top_value_stage3_tf, out_corr_top_index_stage3_tf = tf.nn.top_k(out_corr_score_stage3_tf, k=1024, sorted=True) 
436:def restore_stage2_v2(epoch):
446:def restore_stage3_v1(epoch):
455:def restore_stage3_v2(epoch):
468:def restore_stage3(epoch):
480:  if epoch == 0:
539:      gripper_index = np.array([12])
1232:if __name__ == "__main__":
1240:    restore_stage3(189)
=== point_set_selection_test_with_gt.py
43:nnn = 1
48:parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
49:parser.add_argument('--batch_size',type=int,default=nnn * 1,help='Number of examples within a batch')
64:TOP_K = 1024
65:TOP_K2 = 1024
73:gripper_feat_tf = tf.placeholder(tf.float32,[None,256 * 3])
92:  train_op_nor = tf.train.AdamOptimizer(learning_rate=2 * 1e-3).minimize(loss_nor)
119:  train_op_stage1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage1)
188:  train_op_stage2  = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage2)
400:  train_op_stage3  = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_stage3)
404:  out_corr_top_value_stage3_tf, out_corr_top_index_stage3_tf = tf.nn.top_k(out_corr_score_stage3_tf, k=1024, sorted=True) 
433:def restore_stage2_v2(epoch):
443:def restore_stage3_v1(epoch):
452:def restore_stage3_v2(epoch):
465:def restore_stage3(epoch):
477:  if epoch == 0:
535:      gripper_index = np.array([11])
1223:if __name__ == "__main__":
1224:  restore_stage3(220)
=== point_set_selection_raw_point_cloud.py
42:nnn = 1
47:parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
48:parser.add_argument('--batch_size',type=int,default=nnn * 1,help='Number of examples within a batch')
63:TOP_K = 1024
64:TOP_K2 = 1024
72:gripper_feat_tf = tf.placeholder(tf.float32,[None,256 * 3])
91:  train_op_nor = tf.train.AdamOptimizer(learning_rate=2 * 1e-3).minimize(loss_nor)
118:  train_op_stage1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage1)
187:  train_op_stage2  = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage2)
399:  train_op_stage3  = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_stage3)
403:  out_corr_top_value_stage3_tf, out_corr_top_index_stage3_tf = tf.nn.top_k(out_corr_score_stage3_tf, k=1024, sorted=True) 
432:def restore_stage2_v2(epoch):
442:def restore_stage3_v1(epoch):
451:def restore_stage3_v2(epoch):
464:def restore_stage3(epoch):
476:  if epoch == 0:
499:      gripper_index = np.array([11])
661:if __name__ == "__main__":
662:  restore_stage3(220)
=== unigrasp.py
45:nnn = 1
50:parser.add_argument('--num_epochs',type=int,default=1000 * 1000,help='NUmber of epochs to run trainer')
51:parser.add_argument('--batch_size',type=int,default=nnn * 1,help='Number of examples within a batch')
66:TOP_K = 1024
67:TOP_K2 = 1024
75:gripper_feat_tf = tf.placeholder(tf.float32,[None,256])
94:  train_op_nor = tf.train.AdamOptimizer(learning_rate=2 * 1e-3).minimize(loss_nor)
121:  train_op_stage1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage1)
190:  train_op_stage2  = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_stage2)
402:  train_op_stage3  = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_stage3)
406:  out_corr_top_value_stage3_tf, out_corr_top_index_stage3_tf = tf.nn.top_k(out_corr_score_stage3_tf, k=1024, sorted=True) 
435:def restore_stage2_v2(epoch):
445:def restore_stage3_v1(epoch):
454:def restore_stage3_v2(epoch):
467:def restore_stage3(epoch):
479:  if epoch == 0:
502:      gripper_index = np.array([11])
```
