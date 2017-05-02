/* A c++ version of sparse_predict_client
 * Build it like inception_client.cc
 =======================================================*/
#include <iostream>
#include <fstream>

#include <grpc++/create_channel.h>
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/command_line_flags.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;


using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

typedef google::protobuf::Map< std::string, tensorflow::TensorProto > OutMap;


class ServingClient {
 public:
  ServingClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {
  }
  
  std::string callPredict(std::string model_name) {
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(model_name);

    google::protobuf::Map< std::string, tensorflow::TensorProto >& inputs = 
        *predictRequest.mutable_inputs();

    // Example libSVM data:
    // 0 5:1 6:1 17:1 21:1 35:1 40:1 53:1 63:1 71:1 73:1 74:1 76:1 80:1 83:1
    // 1 5:1 7:1 17:1 22:1 36:1 40:1 51:1 63:1 67:1 73:1 74:1 76:1 81:1 83:1
    
    // Generate keys proto
    tensorflow::TensorProto keys_tensor_proto;
    keys_tensor_proto.set_dtype(tensorflow::DataType::DT_INT32);
    keys_tensor_proto.add_int_val(1);
    keys_tensor_proto.add_int_val(2);
    keys_tensor_proto.mutable_tensor_shape()->add_dim()->set_size(2);
    
    inputs["keys"] = keys_tensor_proto;

    
    // Generate indexs TensorProto
    tensorflow::TensorProto indexs_tensor_proto;
    indexs_tensor_proto.set_dtype(tensorflow::DataType::DT_INT64);
    long indexs[28][2] = { {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5},
                          {0, 6}, {0, 7}, {0, 8}, {0, 9}, {0, 10}, {0, 11},
                          {0, 12}, {0, 13}, {1, 0}, {1, 1}, {1, 2}, {1, 3},
                          {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9},
                          {1, 10}, {1, 11}, {1, 12}, {1, 13} };
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 2; j++) {
            indexs_tensor_proto.add_int64_val(indexs[i][j]);  
        }  
    }
    indexs_tensor_proto.mutable_tensor_shape()->add_dim()->set_size(28);
    indexs_tensor_proto.mutable_tensor_shape()->add_dim()->set_size(2);

    inputs["indexs"] = indexs_tensor_proto;
    std::cout << "Generate indexs tensorproto ok." << std::endl;
  
    // Generate ids TensorProto
    tensorflow::TensorProto ids_tensor_proto;
    ids_tensor_proto.set_dtype(tensorflow::DataType::DT_INT64);
    int ids[28] = {5, 6, 17, 21, 35, 40, 53, 63, 71, 73, 74, 76, 80, 83, 5,
                       7, 17, 22, 36, 40, 51, 63, 67, 73, 74, 76, 81, 83};
    for (int i = 0; i < 28; i++) {
        ids_tensor_proto.add_int64_val(ids[i]); 
    }
    ids_tensor_proto.mutable_tensor_shape()->add_dim()->set_size(28);
    
    inputs["ids"] = ids_tensor_proto;
    std::cout << "Generate ids tensorproto ok." << std::endl;

    // Generate values TensorProto
    tensorflow::TensorProto values_tensor_proto;
    values_tensor_proto.set_dtype(tensorflow::DataType::DT_FLOAT);
    float values[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    for (int i = 0; i < 28; i++) {
        values_tensor_proto.add_float_val(values[i]); 
    }
    values_tensor_proto.mutable_tensor_shape()->add_dim()->set_size(28);

    inputs["values"] = values_tensor_proto;
    std::cout << "Generate values tensorproto ok." << std::endl;

    // Generate shape TensorProto
    tensorflow::TensorProto shape_tensor_proto;
    shape_tensor_proto.set_dtype(tensorflow::DataType::DT_INT64);
    shape_tensor_proto.add_int64_val(2); // ins num
    shape_tensor_proto.add_int64_val(124); // feature num
    shape_tensor_proto.mutable_tensor_shape()->add_dim()->set_size(2);
    
    inputs["shape"] = shape_tensor_proto;
    std::cout << "Generate shape tensorproto ok." << std::endl;

    
    Status status = stub_->Predict(&context, predictRequest, &response);
    
    std::cout << "check status.." << std::endl;

    if (status.ok()) {
      std::cout << "call predict ok" << std::endl;
      std::cout << "outputs size is "<< response.outputs_size() << std::endl;
      OutMap& map_outputs =  *response.mutable_outputs();
      OutMap::iterator iter;
      int output_index = 0;
      
      for(iter = map_outputs.begin();iter != map_outputs.end(); ++iter){
        tensorflow::TensorProto& result_tensor_proto= iter->second;
        tensorflow::Tensor tensor;
        bool converted = tensor.FromProto(result_tensor_proto);
        if (converted) {
          std::cout << "the " <<iter->first <<" result tensor[" << output_index << "] is:" <<
               std::endl << tensor.SummarizeValue(13) << std::endl;
        }else {
          std::cout << "the " <<iter->first <<" result tensor[" << output_index << 
               "] convert failed." << std::endl;
        }
        ++output_index;
      }
      return "Done.";
    } else {
      std::cout << "gRPC call return code: " 
          <<status.error_code() << ": " << status.error_message()
          << std::endl;
      return "gRPC failed.";
    }
  }

 private:
  std::unique_ptr<PredictionService::Stub> stub_;
};

int main(int argc, char** argv) {
  std::string server_port = "localhost:9000";
  std::string model_name = "sparse";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("server_port", &server_port, 
          "the IP and port of the server"),
      tensorflow::Flag("model_name", &model_name, "name of model")
  };
  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cout << usage;
    return -1;
  }

  ServingClient guide(
      grpc::CreateChannel( server_port,
                          grpc::InsecureChannelCredentials()));
  std::cout << "Calling sparse predictor..." << std::endl;
  std::cout << guide.callPredict(model_name) << std::endl;

  return 0;
}
