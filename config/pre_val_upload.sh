#!/bin/sh

#ָ指定输入文件夹，目前只支持文件夹，文件夹下可以有一个或多个文件，
#但只能为同一类型的输入
#更改1
file_path="/home/deepDataBase/bank/dis_shuffle"
if [ ! -d "$file_path" ]
then
	echo "file_path isn't exists"
else
	tmp_output_file="/root/tmp_output"

	#自定义参数
	#更改2
	real_output_dir="/home/deepDataBase/bank/bin_dis_shuffle/"
	txt2proto_param2="0 0"
	txt2proto_param3="20 1 2"
	validate_param="0 0"
	upload_hdfs_dir_prefix="/app/insight/EBG/gushitong/xutao/bank/"
	real_output_train_dir=${real_output_dir}"train/"
	real_output_validate_dir=${real_output_dir}"validate/"
	real_output_test_dir=${real_output_dir}"test/"
	rm -f ${real_output_train_dir}"*"
	rm -f ${real_output_validate_dir}"*"
	rm -f ${real_output_test_dir}"*"

	file_data=${file_path}"/*"
	validate_file=""
	test_file=""
	for single_file in $file_data
	do
		single_file_name=`basename "$single_file"`
		output_str="Txt2protoing "${single_file_name}
		echo -e "\033[32m $output_str \033[0m"
		cat $single_file | txt2proto "$tmp_output_file" "$txt2proto_param2" "$txt2proto_param3" > /dev/null

		if [ ! "$mv_validate" ]
		then
		    real_output_file=${real_output_validate_dir}${single_file_name%.*}".bin"
			mv_validate=1
		elif [ ! "$mv_test" ]
		then
			real_output_file=${real_output_test_dir}${single_file_name%.*}".bin"
			mv_test=1
		else
			real_output_file=${real_output_train_dir}${single_file_name%.*}".bin"

		fi
		echo "$real_output_file"
        mv "$tmp_output_file" "$real_output_file"


		output_str="Validating "${single_file_name}
		echo -e "\033[32m $output_str \033[0m"
		validate $real_output_file $validate_param 2> /dev/null
	done
fi
