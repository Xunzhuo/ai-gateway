// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"encoding/json"
	"fmt"
	"io"
	"path"
	"strconv"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extprocv3 "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
)

// NewEmbeddingOpenAIToOpenAITranslator implements [Factory] for OpenAI to OpenAI translation for embeddings.
func NewEmbeddingOpenAIToOpenAITranslator(apiVersion string, modelNameOverride string) OpenAIEmbeddingTranslator {
	return &openAIToOpenAITranslatorV1Embedding{modelNameOverride: modelNameOverride, path: path.Join("/", apiVersion, "embeddings")}
}

// openAIToOpenAITranslatorV1Embedding implements [OpenAIEmbeddingTranslator] for /embeddings.
type openAIToOpenAITranslatorV1Embedding struct {
	modelNameOverride string
	// The path of the embeddings endpoint to be used for the request. It is prefixed with the OpenAI path prefix.
	path string
	// encodingFormat stores the encoding format from the request to use for response parsing.
	encodingFormat *string
}

// RequestBody implements [OpenAIEmbeddingTranslator.RequestBody].
func (o *openAIToOpenAITranslatorV1Embedding) RequestBody(original []byte, req *openai.EmbeddingRequest, onRetry bool) (
	headerMutation *extprocv3.HeaderMutation, bodyMutation *extprocv3.BodyMutation, err error,
) {
	// Store the encoding format for use in ResponseBody.
	if req != nil {
		o.encodingFormat = req.EncodingFormat
	}
	var newBody []byte
	if o.modelNameOverride != "" {
		// If modelName is set we override the model to be used for the request.
		newBody, err = sjson.SetBytesOptions(original, "model", o.modelNameOverride, SJSONOptions)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to set model name: %w", err)
		}
	}

	// Always set the path header to the embeddings endpoint so that the request is routed correctly.
	headerMutation = &extprocv3.HeaderMutation{
		SetHeaders: []*corev3.HeaderValueOption{
			{Header: &corev3.HeaderValue{
				Key:      ":path",
				RawValue: []byte(o.path),
			}},
		},
	}

	if onRetry && len(newBody) == 0 {
		newBody = original
	}

	if len(newBody) > 0 {
		bodyMutation = &extprocv3.BodyMutation{
			Mutation: &extprocv3.BodyMutation_Body{Body: newBody},
		}
		headerMutation.SetHeaders = append(headerMutation.SetHeaders, &corev3.HeaderValueOption{Header: &corev3.HeaderValue{
			Key:      "content-length",
			RawValue: []byte(strconv.Itoa(len(newBody))),
		}})
	}
	return
}

// ResponseHeaders implements [OpenAIEmbeddingTranslator.ResponseHeaders].
func (o *openAIToOpenAITranslatorV1Embedding) ResponseHeaders(map[string]string) (headerMutation *extprocv3.HeaderMutation, err error) {
	return nil, nil
}

// ResponseBody implements [OpenAIEmbeddingTranslator.ResponseBody].
func (o *openAIToOpenAITranslatorV1Embedding) ResponseBody(_ map[string]string, body io.Reader, _ bool) (
	headerMutation *extprocv3.HeaderMutation, bodyMutation *extprocv3.BodyMutation, tokenUsage LLMTokenUsage, err error,
) {
	// Read the response body.
	bodyBytes, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, tokenUsage, fmt.Errorf("failed to read response body: %w", err)
	}

	// Parse the response with custom embedding handling.
	resp, err := o.parseEmbeddingResponse(bodyBytes)
	if err != nil {
		return nil, nil, tokenUsage, fmt.Errorf("failed to unmarshal body: %w", err)
	}

	tokenUsage = LLMTokenUsage{
		InputTokens: uint32(resp.Usage.PromptTokens), //nolint:gosec
		TotalTokens: uint32(resp.Usage.TotalTokens),  //nolint:gosec
		// Embeddings don't have output tokens, only input and total.
		OutputTokens: 0,
	}
	return
}

// parseEmbeddingResponse parses the embedding response with format-aware embedding parsing.
func (o *openAIToOpenAITranslatorV1Embedding) parseEmbeddingResponse(data []byte) (*openai.EmbeddingResponse, error) {
	// Check if this is an error response first.
	if gjson.GetBytes(data, "error").Exists() {
		// This is an error response, let the standard JSON unmarshaling handle it.
		// The error will be caught by the caller.
		var resp openai.EmbeddingResponse
		if err := json.Unmarshal(data, &resp); err != nil {
			return nil, err
		}
		return &resp, nil
	}

	// Parse the JSON structure manually to handle embeddings with known format.
	result := gjson.GetManyBytes(data, "object", "data", "model", "usage")
	if !result[0].Exists() {
		return nil, fmt.Errorf("invalid response structure")
	}

	resp := &openai.EmbeddingResponse{
		Object: result[0].String(),
		Model:  result[2].String(),
	}

	// Parse usage.
	if result[3].Exists() {
		if err := json.Unmarshal([]byte(result[3].Raw), &resp.Usage); err != nil {
			return nil, fmt.Errorf("failed to parse usage: %w", err)
		}
	}

	// Parse data array with format-aware embedding parsing.
	dataArray := result[1].Array()
	resp.Data = make([]openai.Embedding, len(dataArray))

	for i, item := range dataArray {
		embeddingResult := gjson.GetMany(item.Raw, "object", "embedding", "index")

		resp.Data[i].Object = embeddingResult[0].String()
		resp.Data[i].Index = int(embeddingResult[2].Int())

		// Parse embedding with known format.
		if err := resp.Data[i].Embedding.UnmarshalJSONWithFormat([]byte(embeddingResult[1].Raw), o.encodingFormat); err != nil {
			return nil, fmt.Errorf("failed to parse embedding at index %d: %w", i, err)
		}
	}

	return resp, nil
}

// ResponseError implements [Translator.ResponseError]
// For OpenAI based backend we return the OpenAI error type as is.
// If connection fails the error body is translated to OpenAI error type for events such as HTTP 503 or 504.
func (o *openAIToOpenAITranslatorV1Embedding) ResponseError(respHeaders map[string]string, body io.Reader) (
	headerMutation *extprocv3.HeaderMutation, bodyMutation *extprocv3.BodyMutation, err error,
) {
	statusCode := respHeaders[statusHeaderName]
	if v, ok := respHeaders[contentTypeHeaderName]; ok && v != jsonContentType {
		var openaiError openai.Error
		buf, err := io.ReadAll(body)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to read error body: %w", err)
		}
		openaiError = openai.Error{
			Type: "error",
			Error: openai.ErrorType{
				Type:    openAIBackendError,
				Message: string(buf),
				Code:    &statusCode,
			},
		}
		mut := &extprocv3.BodyMutation_Body{}
		mut.Body, err = json.Marshal(openaiError)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal error body: %w", err)
		}
		headerMutation = &extprocv3.HeaderMutation{}
		setContentLength(headerMutation, mut.Body)
		return headerMutation, &extprocv3.BodyMutation{Mutation: mut}, nil
	}
	return nil, nil, nil
}
