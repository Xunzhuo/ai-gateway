// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
)

func TestOpenAIToOpenAITranslatorV1EmbeddingRequestBody(t *testing.T) {
	for _, tc := range []struct {
		name              string
		modelNameOverride string
		onRetry           bool
		expPath           string
		expBodyContains   string
	}{
		{
			name:            "valid_body",
			expPath:         "/v1/embeddings",
			expBodyContains: "",
		},
		{
			name:              "model_name_override",
			modelNameOverride: "custom-embedding-model",
			expPath:           "/v1/embeddings",
			expBodyContains:   `"model":"custom-embedding-model"`,
		},
		{
			name:    "on_retry",
			onRetry: true,
			expPath: "/v1/embeddings",
		},
		{
			name:              "model_name_override",
			modelNameOverride: "custom-embedding-model",
			onRetry:           true,
			expPath:           "/v1/embeddings",
			expBodyContains:   `"model":"custom-embedding-model"`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator := NewEmbeddingOpenAIToOpenAITranslator("v1", tc.modelNameOverride)
			originalBody := `{"model":"text-embedding-ada-002","input":"test input"}`
			var req openai.EmbeddingRequest
			require.NoError(t, json.Unmarshal([]byte(originalBody), &req))

			headerMutation, bodyMutation, err := translator.RequestBody([]byte(originalBody), &req, tc.onRetry)
			require.NoError(t, err)
			require.NotNil(t, headerMutation)
			require.GreaterOrEqual(t, len(headerMutation.SetHeaders), 1)
			require.Equal(t, ":path", headerMutation.SetHeaders[0].Header.Key)
			require.Equal(t, tc.expPath, string(headerMutation.SetHeaders[0].Header.RawValue))

			switch {
			case tc.expBodyContains != "":
				require.NotNil(t, bodyMutation)
				require.Contains(t, string(bodyMutation.GetBody()), tc.expBodyContains)
				// Verify content-length header is set.
				require.Len(t, headerMutation.SetHeaders, 2)
				require.Equal(t, "content-length", headerMutation.SetHeaders[1].Header.Key)
			case bodyMutation != nil:
				// If there's a body mutation (like on retry), content-length header should be set.
				require.Len(t, headerMutation.SetHeaders, 2)
				require.Equal(t, "content-length", headerMutation.SetHeaders[1].Header.Key)
			default:
				// No body mutation, only path header.
				require.Len(t, headerMutation.SetHeaders, 1)
			}
		})
	}
}

func TestOpenAIToOpenAITranslatorV1EmbeddingResponseHeaders(t *testing.T) {
	translator := NewEmbeddingOpenAIToOpenAITranslator("v1", "")
	headerMutation, err := translator.ResponseHeaders(map[string]string{})
	require.NoError(t, err)
	require.Nil(t, headerMutation)
}

func TestOpenAIToOpenAITranslatorV1EmbeddingResponseBody(t *testing.T) {
	for _, tc := range []struct {
		name                string
		responseBody        string
		responseStatus      string
		expTokenUsage       LLMTokenUsage
		expError            bool
		setupEncodingFormat string // Optional: set encoding format before testing.
	}{
		{
			name: "valid_response",
			responseBody: `{
				"object": "list",
				"data": [
					{
						"object": "embedding",
						"embedding": [0.1, 0.2, 0.3],
						"index": 0
					}
				],
				"model": "text-embedding-ada-002",
				"usage": {
					"prompt_tokens": 8,
					"total_tokens": 8
				}
			}`,
			expTokenUsage: LLMTokenUsage{
				InputTokens:  8,
				OutputTokens: 0,
				TotalTokens:  8,
			},
		},
		{
			name:          "invalid_json",
			responseBody:  `invalid json`,
			expError:      true,
			expTokenUsage: LLMTokenUsage{},
		},
		{
			name: "base64_encoded_embedding",
			responseBody: func() string {
				// Create test data: [0.1, 0.2, 0.3] as float64 in little-endian bytes.
				testFloats := []float64{0.1, 0.2, 0.3}
				var buf bytes.Buffer
				for _, f := range testFloats {
					bits := math.Float64bits(f)
					err := binary.Write(&buf, binary.LittleEndian, bits)
					if err != nil {
						panic(err) // This should never happen in tests.
					}
				}
				base64Str := base64.StdEncoding.EncodeToString(buf.Bytes())

				return fmt.Sprintf(`{
					"object": "list",
					"data": [
						{
							"object": "embedding",
							"embedding": "%s",
							"index": 0
						}
					],
					"model": "text-embedding-ada-002",
					"usage": {
						"prompt_tokens": 5,
						"total_tokens": 5
					}
				}`, base64Str)
			}(),
			expTokenUsage: LLMTokenUsage{
				InputTokens:  5,
				OutputTokens: 0,
				TotalTokens:  5,
			},
			setupEncodingFormat: "base64", // This will be used to set up the translator.
		},
		{
			name:           "error_response",
			responseBody:   `{"error": {"message": "Invalid input", "type": "BadRequestError"}}`,
			responseStatus: "400",
			expTokenUsage:  LLMTokenUsage{},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator := NewEmbeddingOpenAIToOpenAITranslator("v1", "")

			// Set up encoding format if specified.
			if tc.setupEncodingFormat != "" {
				req := &openai.EmbeddingRequest{
					EncodingFormat: &tc.setupEncodingFormat,
				}
				// Call RequestBody to set the encoding format.
				_, _, err := translator.RequestBody([]byte(`{}`), req, false)
				require.NoError(t, err)
			}

			respHeaders := map[string]string{
				"content-type": "application/json",
			}
			if tc.responseStatus != "" {
				respHeaders[statusHeaderName] = tc.responseStatus
			} else {
				respHeaders[statusHeaderName] = "200"
			}

			headerMutation, bodyMutation, tokenUsage, err := translator.ResponseBody(
				respHeaders,
				strings.NewReader(tc.responseBody),
				true,
			)

			if tc.expError {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.Equal(t, tc.expTokenUsage, tokenUsage)

			// Both error and success responses should have nil mutations for OpenAI to OpenAI translation.
			require.Nil(t, headerMutation)
			require.Nil(t, bodyMutation)
		})
	}
}

func TestOpenAIToOpenAITranslatorV1EmbeddingResponseError(t *testing.T) {
	translator := NewEmbeddingOpenAIToOpenAITranslator("v1", "").(*openAIToOpenAITranslatorV1Embedding)

	t.Run("non_json_error", func(t *testing.T) {
		respHeaders := map[string]string{
			statusHeaderName:      "503",
			contentTypeHeaderName: "text/plain",
		}
		errorBody := "Service Unavailable"

		headerMutation, bodyMutation, err := translator.ResponseError(respHeaders, strings.NewReader(errorBody))
		require.NoError(t, err)
		require.NotNil(t, headerMutation)
		require.NotNil(t, bodyMutation)

		// Should convert to OpenAI error format.
		var openaiError openai.Error
		require.NoError(t, json.Unmarshal(bodyMutation.GetBody(), &openaiError))
		require.Equal(t, "error", openaiError.Type)
		require.Equal(t, openAIBackendError, openaiError.Error.Type)
		require.Equal(t, errorBody, openaiError.Error.Message)
		require.Equal(t, "503", *openaiError.Error.Code)
	})

	t.Run("json_error_passthrough", func(t *testing.T) {
		respHeaders := map[string]string{
			statusHeaderName:      "400",
			contentTypeHeaderName: jsonContentType,
		}
		errorBody := `{"error": {"message": "Invalid input", "type": "BadRequestError"}}`

		headerMutation, bodyMutation, err := translator.ResponseError(respHeaders, strings.NewReader(errorBody))
		require.NoError(t, err)
		require.Nil(t, headerMutation)
		require.Nil(t, bodyMutation)
	})
}
