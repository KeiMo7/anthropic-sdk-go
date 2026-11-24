package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/anthropics/anthropic-sdk-go/v2/internal/apierror"
	"github.com/anthropics/anthropic-sdk-go/v2/option"
)

// The fixtures below are local fake servers; no prompt or model output of any
// real request appears in them.

var stableEvents = [][2]string{
	{"message_start", `{"type":"message_start","message":{"id":"msg_resume","type":"message","role":"assistant","model":"claude-resume-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1}}}`},
	{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me "}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think."}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-part-1"}}`},
	{"content_block_stop", `{"type":"content_block_stop","index":0}`},
	{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello, "}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"wor"}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ld!"}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":"Hello","document_index":0,"document_title":"Doc","end_char_index":5,"start_char_index":0}}}`},
	{"content_block_stop", `{"type":"content_block_stop","index":1}`},
	{"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`},
	{"content_block_stop", `{"type":"content_block_stop","index":2}`},
	{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}`},
	{"message_stop", `{"type":"message_stop"}`},
}

var betaEvents = [][2]string{
	{"message_start", `{"type":"message_start","message":{"id":"msg_beta","type":"message","role":"assistant","model":"claude-beta-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}`},
	{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"beta "}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"stream"}}`},
	{"content_block_stop", `{"type":"content_block_stop","index":0}`},
	{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_b","name":"f","input":{}}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}`},
	{"content_block_stop", `{"type":"content_block_stop","index":1}`},
	{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7}}`},
	{"message_stop", `{"type":"message_stop"}`},
}

// streamScript serves one scripted step per request, repeating the last step
// for any further requests.
type streamScript struct {
	mu      sync.Mutex
	attempt int
	steps   []func(w http.ResponseWriter, r *http.Request)
}

func (s *streamScript) server(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		step := s.steps[min(s.attempt, len(s.steps)-1)]
		s.attempt++
		s.mu.Unlock()
		step(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func (s *streamScript) requests() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.attempt
}

func flush(w http.ResponseWriter) {
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// sendEvents writes events as SSE and flushes them.
func sendEvents(events [][2]string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, e := range events {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e[0], e[1])
		}
		flush(w)
	}
}

// thenReset returns a step that runs step and then resets the connection.
func thenReset(step func(http.ResponseWriter, *http.Request)) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		step(w, r)
		resetConnection(w)
	}
}

func resetConnection(w http.ResponseWriter) {
	hj, ok := w.(http.Hijacker)
	if !ok {
		return
	}
	conn, _, err := hj.Hijack()
	if err != nil {
		return
	}
	if tc, ok := conn.(*net.TCPConn); ok {
		tc.SetLinger(0)
	}
	conn.Close()
}

// sendPartial writes events and then the beginning of one more SSE event that
// never completes: the response ends cleanly in the middle of it.
func sendPartial(events [][2]string, partial string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		sendEvents(events)(w, r)
		fmt.Fprint(w, partial)
		flush(w)
	}
}

// sendErrorEvent appends an SSE error event to events.
func sendErrorEvent(events [][2]string, errorType string) func(http.ResponseWriter, *http.Request) {
	return sendEvents(append(slices.Clone(events),
		[2]string{"error", fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":"synthetic test error"}}`, errorType)}))
}

func resumeTestClient(baseURL string) anthropic.Client {
	return anthropic.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey("test-key"),
		option.WithMaxRetries(0),
	)
}

func resumeTestParams() anthropic.MessageNewParams {
	return anthropic.MessageNewParams{
		Model:    anthropic.Model("claude-resume-test"),
		Messages: []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hi"))},
	}
}

func resumeTestConfig() *anthropic.ResumableStreamConfig {
	return &anthropic.ResumableStreamConfig{InitialWait: time.Millisecond, MaxWait: 5 * time.Millisecond}
}

// referenceMessage accumulates events through the existing incremental
// accumulator: the uninterrupted result every spliced stream must reproduce
// byte for byte.
func referenceMessage(t *testing.T, events [][2]string) anthropic.Message {
	t.Helper()
	var msg anthropic.Message
	for _, e := range events {
		var event anthropic.MessageStreamEventUnion
		if err := json.Unmarshal([]byte(e[1]), &event); err != nil {
			t.Fatalf("fixture does not decode: %v", err)
		}
		if err := msg.Accumulate(event); err != nil {
			t.Fatalf("fixture does not accumulate: %v", err)
		}
	}
	return msg
}

func referenceBetaMessage(t *testing.T, events [][2]string) anthropic.BetaMessage {
	t.Helper()
	var msg anthropic.BetaMessage
	for _, e := range events {
		var event anthropic.BetaRawMessageStreamEventUnion
		if err := json.Unmarshal([]byte(e[1]), &event); err != nil {
			t.Fatalf("fixture does not decode: %v", err)
		}
		if err := msg.Accumulate(event); err != nil {
			t.Fatalf("fixture does not accumulate: %v", err)
		}
	}
	return msg
}

// drainEvents reads the stream to its end and returns the delivered event
// types in order.
func drainEvents(stream *anthropic.ResumableMessageStream) []string {
	var types []string
	for stream.Next() {
		types = append(types, stream.Current().Type)
	}
	return types
}

func wantEventTypes(events [][2]string) []string {
	types := make([]string, 0, len(events))
	for _, e := range events {
		types = append(types, e[0])
	}
	return types
}

func TestResumableMessageStreamUninterrupted(t *testing.T) {
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		sendEvents(stableEvents),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	types := drainEvents(stream)
	if err := stream.Err(); err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}
	if !slices.Equal(types, wantEventTypes(stableEvents)) {
		t.Fatalf("delivered events differ:\n got %v\nwant %v", types, wantEventTypes(stableEvents))
	}
	snapshot := stream.Snapshot()
	if !snapshot.Complete || snapshot.Resumed {
		t.Fatalf("snapshot = complete %v resumed %v, want complete and not resumed", snapshot.Complete, snapshot.Resumed)
	}
	if len(snapshot.OpenBlocks) != 0 || len(snapshot.IncompleteToolInputs) != 0 {
		t.Fatalf("snapshot has open blocks %v or incomplete inputs %v", snapshot.OpenBlocks, snapshot.IncompleteToolInputs)
	}
	if got, want := snapshot.Message.RawJSON(), referenceMessage(t, stableEvents).RawJSON(); got != want {
		t.Fatalf("message differs from uninterrupted accumulation:\n got %s\nwant %s", got, want)
	}
	if snapshot.Message.Usage.OutputTokens != 42 {
		t.Fatalf("output tokens = %d, want 42", snapshot.Message.Usage.OutputTokens)
	}
	stats := stream.Stats()
	if stats.Interruptions != 0 || stats.Resumes != 0 || stats.DedupDroppedEvents != 0 || stats.DedupTrimmedEvents != 0 {
		t.Fatalf("uninterrupted stream counted resume activity: %+v", stats)
	}
	if stats.EventsDelivered != uint64(len(stableEvents)) || stats.BytesDelivered == 0 {
		t.Fatalf("delivered counts wrong: %+v", stats)
	}
	if script.requests() != 1 {
		t.Fatalf("requests = %d, want 1", script.requests())
	}
}

func TestResumableMessageStreamConnectionReset(t *testing.T) {
	cut := 9 // through the "wor" text delta
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(stableEvents[:cut])),
		sendEvents(stableEvents),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	types := drainEvents(stream)
	if err := stream.Err(); err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}
	if !slices.Equal(types, wantEventTypes(stableEvents)) {
		t.Fatalf("spliced events differ from uninterrupted stream:\n got %v\nwant %v", types, wantEventTypes(stableEvents))
	}
	snapshot := stream.Snapshot()
	if got, want := snapshot.Message.RawJSON(), referenceMessage(t, stableEvents).RawJSON(); got != want {
		t.Fatalf("spliced message differs from uninterrupted one:\n got %s\nwant %s", got, want)
	}
	if !snapshot.Resumed {
		t.Fatal("snapshot does not report the message was assembled across attempts")
	}
	stats := stream.Stats()
	if stats.Interruptions != 1 || stats.Resumes != 1 {
		t.Fatalf("interruptions/resumes = %d/%d, want 1/1", stats.Interruptions, stats.Resumes)
	}
	if stats.DedupDroppedEvents != uint64(cut) {
		t.Fatalf("dedup dropped = %d, want %d replayed prefix events", stats.DedupDroppedEvents, cut)
	}
	if stats.EventsDelivered != uint64(len(stableEvents)) {
		t.Fatalf("events delivered = %d, want %d", stats.EventsDelivered, len(stableEvents))
	}
	if script.requests() != 2 {
		t.Fatalf("requests = %d, want 2", script.requests())
	}
}

func TestResumableMessageStreamTruncatedRead(t *testing.T) {
	cut := 5
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		sendPartial(stableEvents[:cut], "event: content_block_stop\ndata: {\"type\":\"content_blo"),
		sendEvents(stableEvents),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	types := drainEvents(stream)
	if err := stream.Err(); err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}
	if !slices.Equal(types, wantEventTypes(stableEvents)) {
		t.Fatalf("spliced events differ:\n got %v\nwant %v", types, wantEventTypes(stableEvents))
	}
	snapshot := stream.Snapshot()
	if got, want := snapshot.Message.RawJSON(), referenceMessage(t, stableEvents).RawJSON(); got != want {
		t.Fatalf("spliced message differs:\n got %s\nwant %s", got, want)
	}
	stats := stream.Stats()
	if stats.Interruptions != 1 || stats.Resumes != 1 || stats.DedupDroppedEvents != uint64(cut) {
		t.Fatalf("stats wrong: %+v", stats)
	}
}

func TestResumableMessageStreamStraddlingDelta(t *testing.T) {
	cut := 9 // delivered text so far: "Hello, " + "wor"
	// The resumed stream re-chunkifies the text deltas: the boundary between
	// delivered and new content falls in the middle of a replayed delta.
	rechunked := slices.Clone(stableEvents[:7])
	rechunked = append(rechunked,
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello, w"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"orld!"}}`},
	)
	rechunked = append(rechunked, stableEvents[10:]...)

	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(stableEvents[:cut])),
		sendEvents(rechunked),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	var texts []string
	for stream.Next() {
		event := stream.Current()
		if event.Type == "content_block_delta" && event.Index == 1 && event.Delta.Type == "text_delta" {
			texts = append(texts, event.Delta.Text)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}
	if got := strings.Join(texts, ""); got != "Hello, world!" {
		t.Fatalf("delivered text = %q, want %q (duplicates must be delivered once)", got, "Hello, world!")
	}
	snapshot := stream.Snapshot()
	if got, want := snapshot.Message.RawJSON(), referenceMessage(t, stableEvents).RawJSON(); got != want {
		t.Fatalf("spliced message differs:\n got %s\nwant %s", got, want)
	}
	stats := stream.Stats()
	if stats.DedupTrimmedEvents != 1 {
		t.Fatalf("trimmed events = %d, want 1", stats.DedupTrimmedEvents)
	}
}

func TestResumableMessageStreamDuplicateStopDeliveredOnce(t *testing.T) {
	cut := len(stableEvents) - 1 // everything but message_stop
	// The resumed stream replays the whole prefix and then repeats the
	// termination events: they must arrive exactly once.
	replayed := slices.Clone(stableEvents)
	replayed = append(replayed, stableEvents[len(stableEvents)-2], stableEvents[len(stableEvents)-1])

	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(stableEvents[:cut])),
		sendEvents(replayed),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	types := drainEvents(stream)
	if err := stream.Err(); err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}
	if !slices.Equal(types, wantEventTypes(stableEvents)) {
		t.Fatalf("delivered events differ (termination must be unique):\n got %v\nwant %v", types, wantEventTypes(stableEvents))
	}
	snapshot := stream.Snapshot()
	if got, want := snapshot.Message.RawJSON(), referenceMessage(t, stableEvents).RawJSON(); got != want {
		t.Fatalf("spliced message differs:\n got %s\nwant %s", got, want)
	}
	if snapshot.Message.Usage.OutputTokens != 42 {
		t.Fatalf("output tokens = %d, want 42: usage of repeated attempts must not accumulate", snapshot.Message.Usage.OutputTokens)
	}
	stats := stream.Stats()
	if stats.DedupDroppedEvents != uint64(cut) {
		t.Fatalf("dedup dropped = %d, want %d", stats.DedupDroppedEvents, cut)
	}
}

func TestResumableMessageStreamCompleteBeforeReset(t *testing.T) {
	// The connection resets after message_stop was delivered: the message is
	// complete, so the stream ends cleanly without resuming.
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(stableEvents)),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	types := drainEvents(stream)
	if err := stream.Err(); err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}
	if !slices.Equal(types, wantEventTypes(stableEvents)) {
		t.Fatalf("delivered events differ:\n got %v\nwant %v", types, wantEventTypes(stableEvents))
	}
	stats := stream.Stats()
	if stats.Resumes != 0 {
		t.Fatalf("resumes = %d, want 0: a completed message must not reconnect", stats.Resumes)
	}
	if script.requests() != 1 {
		t.Fatalf("requests = %d, want 1", script.requests())
	}
}

func TestResumableMessageStreamUpstreamErrorEventRetryable(t *testing.T) {
	cut := 6
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		sendErrorEvent(stableEvents[:cut], "overloaded_error"),
		sendEvents(stableEvents),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	types := drainEvents(stream)
	if err := stream.Err(); err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}
	if !slices.Equal(types, wantEventTypes(stableEvents)) {
		t.Fatalf("spliced events differ:\n got %v\nwant %v", types, wantEventTypes(stableEvents))
	}
	stats := stream.Stats()
	if stats.Interruptions != 1 || stats.Resumes != 1 {
		t.Fatalf("interruptions/resumes = %d/%d, want 1/1", stats.Interruptions, stats.Resumes)
	}
}

func TestResumableMessageStreamUpstreamErrorNonRetryable(t *testing.T) {
	cut := 6
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		sendErrorEvent(stableEvents[:cut], "invalid_request_error"),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	drainEvents(stream)
	err := stream.Err()
	var aerr *apierror.Error
	if !errors.As(err, &aerr) {
		t.Fatalf("error = %v (%T), want an *apierror.Error propagated directly", err, err)
	}
	stats := stream.Stats()
	if stats.Resumes != 0 {
		t.Fatalf("resumes = %d, want 0: non-retryable errors must not trigger resumption", stats.Resumes)
	}
	if got := stream.Checkpoint().Events; got != uint64(cut) {
		t.Fatalf("checkpoint advanced to %d events on a non-retryable error, want %d", got, cut)
	}
	if script.requests() != 1 {
		t.Fatalf("requests = %d, want 1", script.requests())
	}
}

func TestResumableMessageStreamInitialRequestUnauthorized(t *testing.T) {
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"type":"error","error":{"type":"authentication_error","message":"synthetic"}}`)
		},
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	drainEvents(stream)
	err := stream.Err()
	var aerr *apierror.Error
	if !errors.As(err, &aerr) || aerr.StatusCode != http.StatusUnauthorized {
		t.Fatalf("error = %v, want the 401 API error", err)
	}
	if stats := stream.Stats(); stats.Resumes != 0 {
		t.Fatalf("resumes = %d, want 0 for rejected credentials", stats.Resumes)
	}
	if script.requests() != 1 {
		t.Fatalf("requests = %d, want 1", script.requests())
	}
}

func TestResumableMessageStreamCheckpointSerializeRestore(t *testing.T) {
	cut := 9
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(stableEvents[:cut])),
		sendEvents(stableEvents),
	}}
	client := resumeTestClient(script.server(t).URL)

	cfg := resumeTestConfig()
	cfg.MaxResumes = -1 // resumption disabled: the interruption surfaces
	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), cfg)
	drainEvents(stream)
	var interrupted *anthropic.StreamInterruptError
	if err := stream.Err(); !errors.As(err, &interrupted) {
		t.Fatalf("error = %v (%T), want *StreamInterruptError", stream.Err(), stream.Err())
	}
	if interrupted.Kind != anthropic.InterruptionConnectionReset {
		t.Fatalf("interruption kind = %q, want %q", interrupted.Kind, anthropic.InterruptionConnectionReset)
	}

	checkpoint := stream.Checkpoint()
	if checkpoint.Events != uint64(cut) {
		t.Fatalf("checkpoint events = %d, want %d", checkpoint.Events, cut)
	}
	if checkpoint.CompletedBlocks != 1 || len(checkpoint.OpenBlocks) != 1 {
		t.Fatalf("checkpoint blocks = %d completed + %d open, want 1+1", checkpoint.CompletedBlocks, len(checkpoint.OpenBlocks))
	}
	open := checkpoint.OpenBlocks[0]
	if open.Index != 1 || open.Type != "text" {
		t.Fatalf("open block = %d/%q, want 1/text", open.Index, open.Type)
	}

	encoded, err := checkpoint.Encode()
	if err != nil {
		t.Fatalf("checkpoint does not encode: %v", err)
	}
	decoded, err := anthropic.DecodeStreamCheckpoint(encoded)
	if err != nil {
		t.Fatalf("checkpoint does not decode: %v", err)
	}
	if !reflect.DeepEqual(decoded, checkpoint) {
		t.Fatalf("checkpoint does not round-trip:\n got %+v\nwant %+v", decoded, checkpoint)
	}

	resumed := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(),
		&anthropic.ResumableStreamConfig{
			Checkpoint:      decoded,
			InitialSnapshot: stream.Snapshot().Message,
			InitialWait:     time.Millisecond,
			MaxWait:         5 * time.Millisecond,
		})
	types := drainEvents(resumed)
	if err := resumed.Err(); err != nil {
		t.Fatalf("resumed stream ended with error: %v", err)
	}
	// Only the continuation is delivered by the restored stream.
	if !slices.Equal(types, wantEventTypes(stableEvents)[cut:]) {
		t.Fatalf("restored stream delivered %v, want the continuation %v", types, wantEventTypes(stableEvents)[cut:])
	}
	snapshot := resumed.Snapshot()
	if !snapshot.MessageAvailable {
		t.Fatal("a seeded restored stream must offer its message")
	}
	if got, want := snapshot.Message.RawJSON(), referenceMessage(t, stableEvents).RawJSON(); got != want {
		t.Fatalf("restored message differs:\n got %s\nwant %s", got, want)
	}
	if !snapshot.Resumed {
		t.Fatal("restored stream does not report it was resumed")
	}
}

func TestResumableMessageStreamCheckpointRestoreSubprocess(t *testing.T) {
	cut := 9
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(stableEvents[:cut])),
		sendEvents(stableEvents),
	}}
	srv := script.server(t)
	client := resumeTestClient(srv.URL)

	cfg := resumeTestConfig()
	cfg.MaxResumes = -1 // resumption disabled: the interruption surfaces
	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), cfg)
	drainEvents(stream)
	if stream.Err() == nil {
		t.Fatal("expected the first attempt to be interrupted")
	}
	encoded, err := stream.Checkpoint().Encode()
	if err != nil {
		t.Fatalf("checkpoint does not encode: %v", err)
	}
	seed, err := json.Marshal(stream.Snapshot().Message)
	if err != nil {
		t.Fatalf("snapshot does not encode: %v", err)
	}
	dir := t.TempDir()
	checkpointFile := filepath.Join(dir, "checkpoint.json")
	seedFile := filepath.Join(dir, "seed.json")
	outputFile := filepath.Join(dir, "message.json")
	if err := os.WriteFile(checkpointFile, encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(seedFile, seed, 0o600); err != nil {
		t.Fatal(err)
	}

	// Continue the stream in a new process from the serialized checkpoint.
	cmd := exec.Command(os.Args[0], "-test.run=^TestResumableStreamHelperProcess$")
	cmd.Env = append(os.Environ(),
		"GO_RESUME_HELPER=1",
		"RESUME_CHECKPOINT_FILE="+checkpointFile,
		"RESUME_SEED_FILE="+seedFile,
		"RESUME_SERVER_URL="+srv.URL,
		"RESUME_OUTPUT_FILE="+outputFile,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("helper process failed: %v\n%s", err, out)
	}
	childRaw, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("helper process wrote no message: %v", err)
	}

	// Continue the same checkpoint in this process, seeded from the same
	// serialized snapshot; both must agree.
	decoded, err := anthropic.DecodeStreamCheckpoint(encoded)
	if err != nil {
		t.Fatal(err)
	}
	var seedMessage anthropic.Message
	if err := json.Unmarshal(seed, &seedMessage); err != nil {
		t.Fatal(err)
	}
	resumed := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(),
		&anthropic.ResumableStreamConfig{
			Checkpoint:      decoded,
			InitialSnapshot: seedMessage,
			InitialWait:     time.Millisecond,
			MaxWait:         5 * time.Millisecond,
		})
	drainEvents(resumed)
	if err := resumed.Err(); err != nil {
		t.Fatalf("in-process resume failed: %v", err)
	}
	inProcessRaw := canonicalMessage(t, resumed.Snapshot().Message)
	if string(childRaw) != inProcessRaw {
		t.Fatalf("cross-process resume differs from in-process resume:\nchild: %s\nlocal: %s", childRaw, inProcessRaw)
	}
	if want := canonicalMessage(t, referenceMessage(t, stableEvents)); inProcessRaw != want {
		t.Fatalf("resumed message differs from uninterrupted one:\n got %s\nwant %s", inProcessRaw, want)
	}
}

// canonicalMessage renders a message in a form that is stable across
// processes, for comparing results that were not accumulated from the same
// event bytes.
func canonicalMessage(t *testing.T, msg anthropic.Message) string {
	t.Helper()
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("message does not encode: %v", err)
	}
	return string(data)
}

// TestResumableStreamHelperProcess is not a test; it is the child process of
// TestResumableMessageStreamCheckpointRestoreSubprocess.
func TestResumableStreamHelperProcess(t *testing.T) {
	if os.Getenv("GO_RESUME_HELPER") != "1" {
		return
	}
	defer os.Exit(0)

	encoded, err := os.ReadFile(os.Getenv("RESUME_CHECKPOINT_FILE"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "helper: %v\n", err)
		os.Exit(1)
	}
	checkpoint, err := anthropic.DecodeStreamCheckpoint(encoded)
	if err != nil {
		fmt.Fprintf(os.Stderr, "helper: %v\n", err)
		os.Exit(1)
	}
	seed, err := os.ReadFile(os.Getenv("RESUME_SEED_FILE"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "helper: %v\n", err)
		os.Exit(1)
	}
	var seedMessage anthropic.Message
	if err := json.Unmarshal(seed, &seedMessage); err != nil {
		fmt.Fprintf(os.Stderr, "helper: %v\n", err)
		os.Exit(1)
	}
	client := resumeTestClient(os.Getenv("RESUME_SERVER_URL"))
	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(),
		&anthropic.ResumableStreamConfig{
			Checkpoint:      checkpoint,
			InitialSnapshot: seedMessage,
			InitialWait:     time.Millisecond,
			MaxWait:         5 * time.Millisecond,
		})
	drainEvents(stream)
	if err := stream.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "helper: stream: %v\n", err)
		os.Exit(1)
	}
	final, err := json.Marshal(stream.Snapshot().Message)
	if err != nil {
		fmt.Fprintf(os.Stderr, "helper: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(os.Getenv("RESUME_OUTPUT_FILE"), final, 0o600); err != nil {
		fmt.Fprintf(os.Stderr, "helper: %v\n", err)
		os.Exit(1)
	}
}

func TestResumableMessageStreamPartialSnapshot(t *testing.T) {
	cut := 14 // through the first input_json delta: the tool input is open
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(stableEvents[:cut])),
		sendEvents(stableEvents),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	for i := 0; i < cut; i++ {
		if !stream.Next() {
			t.Fatalf("stream ended after %d events", i)
		}
	}
	snapshot := stream.Snapshot()
	if snapshot.Complete {
		t.Fatal("snapshot of an interrupted position is complete")
	}
	if !slices.Equal(snapshot.OpenBlocks, []int64{2}) {
		t.Fatalf("open blocks = %v, want [2]", snapshot.OpenBlocks)
	}
	if !slices.Equal(snapshot.IncompleteToolInputs, []int64{2}) {
		t.Fatalf("incomplete tool inputs = %v, want [2]", snapshot.IncompleteToolInputs)
	}
	input := snapshot.Message.Content[2].Input
	if json.Valid(input) {
		t.Fatalf("unclosed tool input %s parses as JSON; it must stay marked incomplete", input)
	}
	if string(input) != `{"location":` {
		t.Fatalf("partial tool input = %s, want the delivered prefix", input)
	}

	// The rest of the stream completes the input.
	for stream.Next() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}
	snapshot = stream.Snapshot()
	if !snapshot.Complete || len(snapshot.OpenBlocks) != 0 || len(snapshot.IncompleteToolInputs) != 0 {
		t.Fatalf("final snapshot = %+v, want complete with no open blocks", snapshot)
	}
	if got := string(snapshot.Message.Content[2].Input); got != `{"location":"Paris"}` {
		t.Fatalf("final tool input = %s", got)
	}
}

func TestResumableMessageStreamBudgetExhausted(t *testing.T) {
	cut := 4
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(stableEvents[:cut])),
		thenReset(sendEvents(stableEvents[:cut])),
		thenReset(sendEvents(stableEvents[:cut])),
		sendEvents(stableEvents),
	}}
	client := resumeTestClient(script.server(t).URL)

	cfg := &anthropic.ResumableStreamConfig{MaxResumes: 2, InitialWait: time.Millisecond, MaxWait: 5 * time.Millisecond}
	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), cfg)
	drainEvents(stream)
	var exhausted *anthropic.ResumeBudgetExhaustedError
	if !errors.As(stream.Err(), &exhausted) {
		t.Fatalf("error = %v (%T), want *ResumeBudgetExhaustedError", stream.Err(), stream.Err())
	}
	if exhausted.Checkpoint.Events != uint64(cut) {
		t.Fatalf("budget error checkpoint = %d events, want %d", exhausted.Checkpoint.Events, cut)
	}
	if exhausted.Stats.Resumes != 2 || exhausted.Stats.Interruptions != 3 {
		t.Fatalf("budget error stats = %+v, want 2 resumes and 3 interruptions", exhausted.Stats)
	}

	// The checkpoint says how far the stream got; a fresh budget continues it.
	resumed := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(),
		&anthropic.ResumableStreamConfig{
			Checkpoint:      exhausted.Checkpoint,
			InitialSnapshot: stream.Snapshot().Message,
			InitialWait:     time.Millisecond,
			MaxWait:         5 * time.Millisecond,
		})
	types := drainEvents(resumed)
	if err := resumed.Err(); err != nil {
		t.Fatalf("stream with a fresh budget did not finish: %v", err)
	}
	if !slices.Equal(types, wantEventTypes(stableEvents)[cut:]) {
		t.Fatalf("continuation delivered %v, want %v", types, wantEventTypes(stableEvents)[cut:])
	}
	if got, want := resumed.Snapshot().Message.RawJSON(), referenceMessage(t, stableEvents).RawJSON(); got != want {
		t.Fatalf("message differs:\n got %s\nwant %s", got, want)
	}
}

func TestResumableMessageStreamInconsistentReplay(t *testing.T) {
	cut := 5
	changed := slices.Clone(stableEvents)
	changed[2] = [2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let us "}}`}
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(stableEvents[:cut])),
		sendEvents(changed),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	drainEvents(stream)
	var inconsistent *anthropic.ResumeInconsistentError
	if !errors.As(stream.Err(), &inconsistent) {
		t.Fatalf("error = %v (%T), want *ResumeInconsistentError", stream.Err(), stream.Err())
	}
	if inconsistent.Checkpoint.Events != uint64(cut) {
		t.Fatalf("inconsistency checkpoint = %d events, want %d", inconsistent.Checkpoint.Events, cut)
	}
	if script.requests() != 2 {
		t.Fatalf("requests = %d, want 2: an inconsistent stream must not be retried further", script.requests())
	}
}

func TestResumableMessageStreamCallerCancel(t *testing.T) {
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		func(w http.ResponseWriter, r *http.Request) {
			sendEvents(stableEvents[:3])(w, r)
			<-r.Context().Done()
		},
	}}
	client := resumeTestClient(script.server(t).URL)

	ctx, cancel := context.WithCancel(context.Background())
	stream := client.Messages.NewResumableStreaming(ctx, resumeTestParams(), resumeTestConfig())
	for i := 0; i < 3; i++ {
		if !stream.Next() {
			t.Fatalf("stream ended after %d events", i)
		}
	}
	cancel()
	if stream.Next() {
		t.Fatal("stream delivered an event after cancellation")
	}
	err := stream.Err()
	var interrupted *anthropic.StreamInterruptError
	if !errors.As(err, &interrupted) {
		t.Fatalf("error = %v (%T), want *StreamInterruptError", err, err)
	}
	if interrupted.Kind != anthropic.InterruptionCallerCancel {
		t.Fatalf("interruption kind = %q, want %q", interrupted.Kind, anthropic.InterruptionCallerCancel)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error %v does not unwrap to context.Canceled", err)
	}
	time.Sleep(50 * time.Millisecond)
	if script.requests() != 1 {
		t.Fatalf("requests = %d, want 1: a canceled stream must not resume", script.requests())
	}
}

func TestResumableMessageStreamCallerTimeout(t *testing.T) {
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		func(w http.ResponseWriter, r *http.Request) {
			sendEvents(stableEvents[:3])(w, r)
			<-r.Context().Done()
		},
	}}
	client := resumeTestClient(script.server(t).URL)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	stream := client.Messages.NewResumableStreaming(ctx, resumeTestParams(), resumeTestConfig())
	drainEvents(stream)
	err := stream.Err()
	var interrupted *anthropic.StreamInterruptError
	if !errors.As(err, &interrupted) {
		t.Fatalf("error = %v (%T), want *StreamInterruptError", err, err)
	}
	if interrupted.Kind != anthropic.InterruptionCallerTimeout {
		t.Fatalf("interruption kind = %q, want %q", interrupted.Kind, anthropic.InterruptionCallerTimeout)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error %v does not unwrap to context.DeadlineExceeded", err)
	}
	time.Sleep(50 * time.Millisecond)
	if script.requests() != 1 {
		t.Fatalf("requests = %d, want 1: a timed-out stream must not resume", script.requests())
	}
}

func TestResumableMessageStreamCheckpointFromAnotherRequest(t *testing.T) {
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		sendEvents(stableEvents),
	}}
	client := resumeTestClient(script.server(t).URL)

	stream := client.Messages.NewResumableStreaming(context.Background(), resumeTestParams(), resumeTestConfig())
	if !stream.Next() {
		t.Fatal("stream delivered nothing")
	}
	checkpoint := stream.Checkpoint()

	otherParams := resumeTestParams()
	otherParams.MaxTokens = 2048
	resumed := client.Messages.NewResumableStreaming(context.Background(), otherParams,
		&anthropic.ResumableStreamConfig{Checkpoint: checkpoint})
	if resumed.Next() {
		t.Fatal("a checkpoint of another request resumed")
	}
	var impossible *anthropic.ResumeImpossibleError
	if !errors.As(resumed.Err(), &impossible) {
		t.Fatalf("error = %v (%T), want *ResumeImpossibleError", resumed.Err(), resumed.Err())
	}
}

func TestBetaResumableMessageStreamConnectionReset(t *testing.T) {
	cut := 4
	script := &streamScript{steps: []func(http.ResponseWriter, *http.Request){
		thenReset(sendEvents(betaEvents[:cut])),
		sendEvents(betaEvents),
	}}
	client := resumeTestClient(script.server(t).URL)

	params := anthropic.BetaMessageNewParams{
		Model:    anthropic.Model("claude-beta-test"),
		Messages: []anthropic.BetaMessageParam{anthropic.NewBetaUserMessage(anthropic.NewBetaTextBlock("hi"))},
	}
	stream := client.Beta.Messages.NewResumableStreaming(context.Background(), params, resumeTestConfig())
	var types []string
	for stream.Next() {
		types = append(types, stream.Current().Type)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}
	if !slices.Equal(types, wantEventTypes(betaEvents)) {
		t.Fatalf("spliced beta events differ:\n got %v\nwant %v", types, wantEventTypes(betaEvents))
	}
	snapshot := stream.Snapshot()
	if got, want := snapshot.Message.RawJSON(), referenceBetaMessage(t, betaEvents).RawJSON(); got != want {
		t.Fatalf("spliced beta message differs:\n got %s\nwant %s", got, want)
	}
	stats := stream.Stats()
	if stats.Interruptions != 1 || stats.Resumes != 1 || stats.DedupDroppedEvents != uint64(cut) {
		t.Fatalf("stats wrong: %+v", stats)
	}
	if !snapshot.Resumed || !snapshot.Complete {
		t.Fatalf("snapshot = resumed %v complete %v", snapshot.Resumed, snapshot.Complete)
	}
}
