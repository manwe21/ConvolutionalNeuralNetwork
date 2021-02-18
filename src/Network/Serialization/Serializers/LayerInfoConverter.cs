using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Network.Serialization.Serializers
{
    public class LayerInfoConverter : JsonConverter<LayerInfo>
    {
        public override bool CanConvert(Type typeToConvert)
        {
            return typeof(LayerInfo).IsAssignableFrom(typeToConvert);
        }

        public override LayerInfo Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();
            if (!reader.Read() || reader.TokenType != JsonTokenType.PropertyName || reader.GetString() != nameof(LayerDiscriminator))
                throw new JsonException();
            if (!reader.Read() || reader.TokenType != JsonTokenType.Number)
                throw new JsonException();

            LayerDiscriminator layerDiscriminator = (LayerDiscriminator)reader.GetInt32();
            
            if (!reader.Read() || reader.GetString() != "LayerData")
                throw new JsonException();
            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();
            
            var type = layerDiscriminator.GetLayerInfoType();
            var layerInfo = (LayerInfo)JsonSerializer.Deserialize(ref reader, type);

            if (!reader.Read() || reader.TokenType != JsonTokenType.EndObject)
                throw new JsonException();
            
            return layerInfo;
        }
        
        //looks bad
        public override void Write(Utf8JsonWriter writer, LayerInfo value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            
            if (value is ConvolutionLayerInfo convolutionLayerInfo)
            {
                writer.WriteNumber(nameof(LayerDiscriminator), (int)LayerDiscriminator.ConvolutionLayerInfo);
                writer.WritePropertyName("LayerData");
                JsonSerializer.Serialize(writer, convolutionLayerInfo);
            }
            else if (value is ActivationLayerInfo activationLayerInfo)
            {
                writer.WriteNumber(nameof(LayerDiscriminator), (int)LayerDiscriminator.ActivationLayerInfo);
                writer.WritePropertyName("LayerData");
                JsonSerializer.Serialize(writer, activationLayerInfo);
            }
            else if (value is ParameterizedLayerInfo parameterizedLayerInfo)
            {
                writer.WriteNumber(nameof(LayerDiscriminator), (int)LayerDiscriminator.ParameterizedLayerInfo);
                writer.WritePropertyName("LayerData");
                JsonSerializer.Serialize(writer, parameterizedLayerInfo);
            }
            else if (value is PoolingLayerInfo poolingLayerInfo)
            {
                writer.WriteNumber(nameof(LayerDiscriminator), (int)LayerDiscriminator.PoolingLayerInfo);
                writer.WritePropertyName("LayerData");
                JsonSerializer.Serialize(writer, poolingLayerInfo);
            }
            else if (value is PaddingLayerInfo paddingLayerInfo)
            {
                writer.WriteNumber(nameof(LayerDiscriminator), (int)LayerDiscriminator.PaddingLayerInfo);
                writer.WritePropertyName("LayerData");
                JsonSerializer.Serialize(writer, paddingLayerInfo);
            }
            else
            {
                writer.WriteNumber(nameof(LayerDiscriminator), (int)LayerDiscriminator.LayerInfo);
                writer.WritePropertyName("LayerData");
                JsonSerializer.Serialize(writer, value);
            }

            writer.WriteEndObject();
        }
    }
}
