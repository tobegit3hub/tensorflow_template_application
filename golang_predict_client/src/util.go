package main

import (
	"errors"
	"reflect"
	framework "tensorflow/core/framework"
	pb "tensorflow_serving"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
)

func newPredictRequest(modelName string, modelVersion int64) (pr *pb.PredictRequest) {
	return &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name: modelName,
			Version: &google_protobuf.Int64Value{
				Value: modelVersion,
			},
		},
		Inputs: make(map[string]*framework.TensorProto),
	}
}

// if tensor is one dim, shapeSize is nil
func addInput(pr *pb.PredictRequest, tensorName string, dataType framework.DataType, tensor interface{},
	shapeSize []int64, shapeName []string) (err error) {
	v := reflect.ValueOf(tensor)
	if v.Kind() != reflect.Slice {
		return errors.New("tensor must be slice")
	}
	size := v.Len()
	tp := &framework.TensorProto{
		Dtype: dataType,
	}

	var ok bool
	switch dataType {
	case framework.DataType_DT_HALF:
		tp.HalfVal, ok = tensor.([]int32)
	case framework.DataType_DT_FLOAT:
		tp.FloatVal, ok = tensor.([]float32)
	case framework.DataType_DT_DOUBLE:
		tp.DoubleVal, ok = tensor.([]float64)
	case framework.DataType_DT_INT16, framework.DataType_DT_INT32,
		framework.DataType_DT_INT8, framework.DataType_DT_UINT8:
		tp.IntVal, ok = tensor.([]int32)
	case framework.DataType_DT_STRING:
		tp.StringVal, ok = tensor.([][]byte)
	case framework.DataType_DT_COMPLEX64:
		tp.ScomplexVal, ok = tensor.([]float32)
	case framework.DataType_DT_INT64:
		tp.Int64Val, ok = tensor.([]int64)
	case framework.DataType_DT_BOOL:
		tp.BoolVal, ok = tensor.([]bool)
	case framework.DataType_DT_COMPLEX128:
		tp.DcomplexVal, ok = tensor.([]float64)
	case framework.DataType_DT_RESOURCE:
		tp.ResourceHandleVal, ok = tensor.([]*framework.ResourceHandle)
	default:
		err = errors.New("Unknown data type")
	}

	if !ok {
		if err != nil {
			err = errors.New("Type switch failed")
		}
		return
	}

	if shapeSize == nil {
		name := ""
		if len(shapeName) != 0 {
			name = shapeName[0]
		}
		tp.TensorShape = &framework.TensorShapeProto{
			Dim: []*framework.TensorShapeProto_Dim{
				&framework.TensorShapeProto_Dim{
					Size: int64(size),
					Name: name,
				},
			},
		}
	} else {
		if shapeName != nil && len(shapeName) != len(shapeSize) {
			return errors.New("shapeName and shapeSize have different size")
		}
		tp.TensorShape = &framework.TensorShapeProto{
			Dim: []*framework.TensorShapeProto_Dim{},
		}
		for i, size := range shapeSize {
			name := ""
			if shapeName != nil {
				name = shapeName[i]
			}
			tp.TensorShape.Dim = append(tp.TensorShape.Dim, &framework.TensorShapeProto_Dim{
				Size: size,
				Name: name,
			})
		}
	}
	pr.Inputs[tensorName] = tp
	return
}
