#!/bin/sh

#ָ指定输入文件夹，目前只支持文件夹，文件夹下可以有一个或多个文件，
#但只能为同一类型的输入
#更改1
file_path="/home/zz800_feature/vis-40-20-g20"
if [ ! -d "$file_path" ]
then
	echo "file_path isn't exists"
else
	tmp_output_file="/root/tmp_output"

	#自定义参数
	#更改2
	real_output_dir="/home/zz800_feature/tmp"
	txt2proto_param2="0 0"
	txt2proto_param3="20 1 2"
	validate_param="0 0"
	#更改3
	upload_hdfs_dir_prefix="/app/insight/EBG/gushitong/xutao/000001same/"
	upload_hdfs_train_dir=${upload_hdfs_dir_prefix}"train/data/"
	upload_hdfs_validate_dir=${upload_hdfs_dir_prefix}"validate/data/"
	upload_hdfs_test_dir=${upload_hdfs_dir_prefix}"test/data/"
	hadoop fs -rm ${upload_hdfs_train_dir}"*"
	hadoop fs -rm ${upload_hdfs_validate_dir}"*"
	hadoop fs -rm ${upload_hdfs_test_dir}"*"

	file_data=${file_path}"/*"
	validate_file=""
	test_file=""
	for single_file in $file_data
	do
		single_file_name=`basename "$single_file"`
		output_str="Txt2protoing "${single_file_name}
		echo -e "\033[32m $output_str \033[0m"
		cat $single_file | txt2proto "$tmp_output_file" "$txt2proto_param2" "$txt2proto_param3" > /dev/null
		if [ ! -d "$real_output_dir" ]
		then
			mkdir "$real_output_dir"
		fi
		real_output_file=${real_output_dir}"/"${single_file_name%.*}".bin"
		mv "$tmp_output_file" "$real_output_file"

		output_str="Validating "${single_file_name}
		echo -e "\033[32m $output_str \033[0m"
		validate $real_output_file $validate_param 2> /dev/null
		if [ ! "$upload_validate" ]
		then
			hadoop fs -put $real_output_file $upload_hdfs_validate_dir
			upload_validate=1
		elif [ ! "$upload_test" ]
		then
			hadoop fs -put $real_output_file $upload_hdfs_test_dir
			upload_test=1
		else
			hadoop fs -put $real_output_file $upload_hdfs_train_dir
		fi
	done
fi
