�	sh���!�@sh���!�@!sh���!�@	�u�;;�?�u�;;�?!�u�;;�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$sh���!�@�$��C�?Aj�t�@Y����S�?*	     �}@2F
Iterator::ModelL7�A`��?!���R�T@)��x�&1�?1���&��S@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�v��/�?!�JG�(@)�A`��"�?1z���\&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/�$��?!q�w��@)����Mb�?1����@:Preprocessing2U
Iterator::Model::ParallelMapV2�I+��?!w���@)�I+��?1w���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��(\�µ?!3���1@)y�&1�|?1��2���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zt?!l��Ӭ��?){�G�zt?1l��Ӭ��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!�����?)����Mbp?1�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�u�;;�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�$��C�?�$��C�?!�$��C�?      ��!       "      ��!       *      ��!       2	j�t�@j�t�@!j�t�@:      ��!       B      ��!       J	����S�?����S�?!����S�?R      ��!       Z	����S�?����S�?!����S�?JCPU_ONLYY�u�;;�?b Y      Y@q�4��Lp?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 