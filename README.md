# BankNameMatch
数据治理银行名称标准化
# 接口调用参数
- /v1/process_and_search/
  - 入参:
    
    |参数|含义|类型|
    |----|----|----|
    |query_file|上传的文件|UploadFile文件对象|

  - 返回：

    |参数|含义|类型|
    |----|----|----|
    | unique_id|文件唯一id标识|string|

- /check_status/
  - 入参:
 
    |参数|含义|类型|
    |----|----|----|
    |unique_id|文件唯一id标识|string|

  - 返回：

    |参数|类型|含义
    |----|----|----
    |unique_id|文件唯一id标识|string
    |status|文件状态:processing、completed、error|string
    |result_file|结果文件路径|string

- /download_result/
  - 入参:

    |参数|含义|类型|
    |----|----|----|
    |unique_id|文件唯一id标识|string|
    
  - 返回:
 
    |参数|含义|类型|
    |----|----|----|
    |-|下载文件对象|FileResponse对象|
