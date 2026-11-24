package anthropic

// Resumable streaming for the Messages APIs.
//
// A resumable stream wraps the event stream of [MessageService.NewStreaming]
// (or its beta) with a reconnect-and-splice layer: when the connection drops
// mid-stream the stream re-issues the same request, verifies that the replayed
// prefix matches what was already delivered, drops the duplicate events and
// delivers only the continuation, so the application sees one continuous
// stream. The position reached so far is captured by a [StreamCheckpoint],
// which can be serialized and restored in another process to continue the
// stream there.
//
// The layer is opt-in: [MessageService.NewStreaming] is unchanged, and a
// resumable stream with MaxResumes == 0 makes no reconnections. Decoding and
// incremental accumulation reuse the existing [ssestream] decoder and
// [Message.Accumulate]; no event is parsed twice and no duplicate is
// accumulated, so usage counts and termination events are delivered exactly
// once.

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"net"
	"net/http"
	"slices"
	"strconv"
	"time"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/anthropics/anthropic-sdk-go/v2/internal/apierror"
	"github.com/anthropics/anthropic-sdk-go/v2/option"
	"github.com/anthropics/anthropic-sdk-go/v2/packages/ssestream"
	"github.com/anthropics/anthropic-sdk-go/v2/shared"
)

// ResumableStreamConfig configures a stream created by
// [MessageService.NewResumableStreaming] or
// [BetaMessageService.NewResumableStreaming]. The zero value is valid and
// applies the defaults documented per field; passing a nil *Config to the
// constructor is equivalent to the zero value.
type ResumableStreamConfig struct {
	// Checkpoint resumes the stream from a previously captured
	// [StreamCheckpoint], possibly written by another process. The zero value
	// starts a fresh stream.
	Checkpoint StreamCheckpoint
	// InitialSnapshot seeds the stream's partial result with the
	// [MessageStreamSnapshot.Message] (or [BetaMessageStreamSnapshot.Message])
	// of the stream the Checkpoint came from, so that Snapshot of the
	// restored stream covers the whole message. It must be a Message or
	// *Message (BetaMessage or *BetaMessage for the beta API). A checkpoint
	// restores fine without it, but Snapshot then cannot offer the message
	// content that predates the checkpoint and reports MessageAvailable ==
	// false.
	InitialSnapshot any
	// MaxResumes bounds how many times the stream reconnects after an
	// interruption. 0 applies the default of 5; a negative value disables
	// resumption: the first interruption ends the stream with an error.
	MaxResumes int
	// InitialWait is the wait before the first resume attempt; later attempts
	// back off exponentially up to MaxWait. A Retry-After hint from the API
	// overrides the computed wait. Defaults to 500ms.
	InitialWait time.Duration
	// MaxWait caps the wait between resume attempts. Defaults to 10s.
	MaxWait time.Duration
}

const (
	resumeDefaultMaxResumes  = 5
	resumeDefaultInitialWait = 500 * time.Millisecond
	resumeDefaultMaxWait     = 10 * time.Second
)

func (c *ResumableStreamConfig) normalized() ResumableStreamConfig {
	if c == nil {
		return ResumableStreamConfig{
			MaxResumes:  resumeDefaultMaxResumes,
			InitialWait: resumeDefaultInitialWait,
			MaxWait:     resumeDefaultMaxWait,
		}
	}
	out := *c
	if out.MaxResumes == 0 {
		out.MaxResumes = resumeDefaultMaxResumes
	}
	if out.MaxResumes < 0 {
		out.MaxResumes = 0
	}
	if out.InitialWait <= 0 {
		out.InitialWait = resumeDefaultInitialWait
	}
	if out.MaxWait <= 0 {
		out.MaxWait = resumeDefaultMaxWait
	}
	return out
}

// StreamCheckpoint marks the position a resumable stream has delivered up to:
// how far each content block has progressed and the offsets within it. It is
// serializable; a checkpoint written by one process can be restored in
// another (via [ResumableStreamConfig.Checkpoint]) and the stream continues
// from the same position.
//
// A checkpoint advances only when an event is delivered, never ahead of
// delivered content. Its memory footprint is bounded by the number of
// currently open content blocks, not by the length of the output.
type StreamCheckpoint struct {
	Version int `json:"version"`
	// RequestHash identifies the request the checkpoint belongs to; restoring
	// it against different parameters is rejected. It is a hash of the
	// serialized parameters, never the parameters themselves.
	RequestHash uint64 `json:"request_hash,omitempty"`
	// Events and Bytes count the events delivered so far and their wire bytes.
	Events uint64 `json:"events"`
	Bytes  uint64 `json:"bytes"`
	// Resumed reports whether the stream was assembled across more than one
	// attempt or restored from a checkpoint.
	Resumed        bool   `json:"resumed,omitempty"`
	MessageStarted bool   `json:"message_started,omitempty"`
	MessageDeltas  uint64 `json:"message_deltas,omitempty"`
	MessageStopped bool   `json:"message_stopped,omitempty"`
	// CompletedBlocks and CompletedDigest summarize the content blocks whose
	// stop event was delivered, folded into one rolling digest.
	CompletedBlocks uint64 `json:"completed_blocks,omitempty"`
	CompletedDigest uint64 `json:"completed_digest,omitempty"`
	// OpenBlocks holds the blocks whose stop event has not been delivered yet.
	OpenBlocks []BlockCheckpoint `json:"open_blocks,omitempty"`
}

// Encode serializes the checkpoint to JSON.
func (c StreamCheckpoint) Encode() ([]byte, error) {
	return json.Marshal(c)
}

// DecodeStreamCheckpoint restores a checkpoint serialized with
// [StreamCheckpoint.Encode].
func DecodeStreamCheckpoint(data []byte) (StreamCheckpoint, error) {
	var c StreamCheckpoint
	if err := json.Unmarshal(data, &c); err != nil {
		return StreamCheckpoint{}, fmt.Errorf("anthropic: cannot decode stream checkpoint: %w", err)
	}
	if err := c.validate(); err != nil {
		return StreamCheckpoint{}, err
	}
	return c, nil
}

func (c StreamCheckpoint) validate() error {
	if c.Version != 0 && c.Version != streamCheckpointVersion {
		return &ResumeImpossibleError{
			Reason:     fmt.Sprintf("checkpoint version %d is not supported by this SDK", c.Version),
			Checkpoint: c,
		}
	}
	seen := map[int64]bool{}
	for _, b := range c.OpenBlocks {
		if b.Index < 0 {
			return &ResumeImpossibleError{Reason: "checkpoint has a content block with a negative index", Checkpoint: c}
		}
		if seen[b.Index] {
			return &ResumeImpossibleError{Reason: "checkpoint lists a content block twice", Checkpoint: c}
		}
		seen[b.Index] = true
	}
	return nil
}

// IsZero reports whether the checkpoint marks the start of a stream.
func (c StreamCheckpoint) IsZero() bool {
	return c.Events == 0 && !c.MessageStarted && len(c.OpenBlocks) == 0 && c.CompletedBlocks == 0
}

// BlockCheckpoint is the position within one not-yet-stopped content block.
type BlockCheckpoint struct {
	Index int64  `json:"index"`
	Type  string `json:"type"`
	// Streams holds one mark per delta stream the block received: text,
	// thinking, signature, input_json, citations, and any other delta type.
	Streams []DeltaStreamMark `json:"streams,omitempty"`
}

// DeltaStreamMark is the delivered length and rolling content digest of one
// delta stream of a content block. For text-like streams Len counts bytes;
// for discrete deltas such as citations it counts items.
type DeltaStreamMark struct {
	Name string `json:"name"`
	Len  uint64 `json:"len"`
	Hash uint64 `json:"hash"`
}

const streamCheckpointVersion = 1

// ResumeStats reports what a resumable stream has done, in the same spirit as
// the SDK's OpenTelemetry instrumentation: counters only, no content. It is
// purely additive; streams that never interrupt report zeros in every
// resume-specific field.
type ResumeStats struct {
	// Interruptions counts how often an attempt ended before message_stop.
	Interruptions uint64 `json:"interruptions"`
	// Resumes counts reconnections actually performed.
	Resumes uint64 `json:"resumes"`
	// EventsDelivered and BytesDelivered count the events handed to the
	// application and their wire bytes. They continue cumulatively when a
	// stream is restored from a checkpoint.
	EventsDelivered uint64 `json:"events_delivered"`
	BytesDelivered  uint64 `json:"bytes_delivered"`
	// DedupDroppedEvents and DedupDroppedBytes count replayed events (and
	// replayed bytes) recognized as duplicates of already-delivered content
	// and dropped instead of delivered.
	DedupDroppedEvents uint64 `json:"dedup_dropped_events"`
	DedupDroppedBytes  uint64 `json:"dedup_dropped_bytes"`
	// DedupTrimmedEvents counts delta events that overlapped the delivered
	// prefix and were delivered with the duplicate head trimmed off.
	DedupTrimmedEvents uint64 `json:"dedup_trimmed_events"`
}

// InterruptionKind classifies how a stream attempt ended.
type InterruptionKind string

const (
	// InterruptionConnectionReset is a transport failure: the connection was
	// reset, broke, or timed out at the network level.
	InterruptionConnectionReset InterruptionKind = "connection_reset"
	// InterruptionTruncated is a response that ended cleanly but before
	// message_stop: the read stopped mid-stream.
	InterruptionTruncated InterruptionKind = "truncated"
	// InterruptionUpstreamError is an error reported by the API, either as a
	// streaming error event or as an error response.
	InterruptionUpstreamError InterruptionKind = "upstream_error"
	// InterruptionCallerCancel is cancellation of the caller's context. A
	// stream interrupted this way is never resumed.
	InterruptionCallerCancel InterruptionKind = "caller_cancel"
	// InterruptionCallerTimeout is expiry of the caller's context deadline. A
	// stream interrupted this way is never resumed.
	InterruptionCallerTimeout InterruptionKind = "caller_timeout"
	// InterruptionAttemptTimeout is expiry of a per-attempt timeout set by
	// the request options while the caller's context is still alive.
	InterruptionAttemptTimeout InterruptionKind = "attempt_timeout"
	// InterruptionUnknown is any other failure; it is not resumed.
	InterruptionUnknown InterruptionKind = "unknown"
)

// StreamInterruptError ends a resumable stream when an interruption cannot be
// or may not be resumed (for example resumption is disabled, or the caller
// canceled). It carries the kind of interruption and the checkpoint of the
// position delivered so far; the stream never ends this way silently, it
// always reports the position it stopped at.
type StreamInterruptError struct {
	Kind       InterruptionKind
	Checkpoint StreamCheckpoint
	Err        error
}

func (e *StreamInterruptError) Error() string {
	return fmt.Sprintf("anthropic: stream interrupted (%s) after %d delivered events: %v", e.Kind, e.Checkpoint.Events, e.Err)
}

func (e *StreamInterruptError) Unwrap() error { return e.Err }

// ResumeBudgetExhaustedError ends a resumable stream when the interruption
// could be resumed but the resume budget (MaxResumes) is spent. The
// checkpoint says exactly how far the stream was consumed; a new stream
// created from it with a fresh budget can continue.
type ResumeBudgetExhaustedError struct {
	Checkpoint StreamCheckpoint
	Stats      ResumeStats
	Err        error
}

func (e *ResumeBudgetExhaustedError) Error() string {
	return fmt.Sprintf("anthropic: stream resume budget exhausted after %d resumes with %d events delivered: %v",
		e.Stats.Resumes, e.Checkpoint.Events, e.Err)
}

func (e *ResumeBudgetExhaustedError) Unwrap() error { return e.Err }

// ResumeInconsistentError ends a resumable stream when the resumed stream
// contradicts what was already delivered, beyond the consistency tolerance
// (message metadata and usage values are not compared; content is). The
// mismatched content is never spliced into the result. Detail names the
// entity that mismatched; it never contains prompt or model output.
type ResumeInconsistentError struct {
	Checkpoint StreamCheckpoint
	Detail     string
}

func (e *ResumeInconsistentError) Error() string {
	return fmt.Sprintf("anthropic: resumed stream is inconsistent with the content already delivered: %s", e.Detail)
}

// ResumeImpossibleError is reported when a stream cannot be resumed at all:
// the checkpoint does not match the request, or the checkpoint itself is not
// usable. It carries the checkpoint that was rejected.
type ResumeImpossibleError struct {
	Reason     string
	Checkpoint StreamCheckpoint
}

func (e *ResumeImpossibleError) Error() string {
	return "anthropic: stream cannot be resumed: " + e.Reason
}

// fnv-1a 64, kept as a plain uint64 chain so a running hash is exactly its
// serializable state: a checkpoint stores the current value and another
// process continues the chain from it.
const (
	fnvOffset64 = 14695981039346656037
	fnvPrime64  = 1099511628211
)

func fnv1a(h uint64, p []byte) uint64 {
	for _, b := range p {
		h ^= uint64(b)
		h *= fnvPrime64
	}
	return h
}

func fnv1aString(h uint64, s string) uint64 { return fnv1a(h, []byte(s)) }

func fnv1aUint(h, v uint64) uint64 {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	return fnv1a(h, b[:])
}

// deltaKind distinguishes delta streams that are byte strings re-chunkable by
// a replay (text, thinking, ...) from deltas that arrive as discrete items
// (citations and any unmodeled delta type).
type deltaKind int

const (
	deltaBytes deltaKind = iota
	deltaItems
)

// streamDelta is one delta's contribution to a content block: which stream of
// the block it extends, and with what payload.
type streamDelta struct {
	stream  string
	kind    deltaKind
	payload string
}

// eventInfo is what the resume engine needs to know about one decoded event.
type eventInfo struct {
	typ   string
	index int64
	// blockType is the content block's type, for content_block_start.
	blockType string
	// delta is the delta of a content_block_delta event, nil for events
	// without one.
	delta *streamDelta
	// raw is the event's wire JSON.
	raw string
}

// deltaStreamState is the running length and content digest of one delta
// stream of a content block.
type deltaStreamState struct {
	length uint64
	hash   uint64
}

func newDeltaStreamState() *deltaStreamState {
	return &deltaStreamState{hash: fnvOffset64}
}

func (s *deltaStreamState) feed(d streamDelta) { s.feedBytes(d.kind, d.payload) }

func (s *deltaStreamState) feedBytes(kind deltaKind, payload string) {
	if kind == deltaBytes {
		s.hash = fnv1aString(s.hash, payload)
		s.length += uint64(len(payload))
		return
	}
	// Discrete items are framed with their length so that item boundaries
	// take part in the digest.
	s.hash = fnv1aUint(s.hash, uint64(len(payload)))
	s.hash = fnv1aString(s.hash, payload)
	s.length++
}

func (s *deltaStreamState) clone() *deltaStreamState {
	c := *s
	return &c
}

// blockState tracks the delta streams of one content block.
type blockState struct {
	typ     string
	streams map[string]*deltaStreamState
}

func newBlockState(typ string) *blockState {
	return &blockState{typ: typ, streams: map[string]*deltaStreamState{}}
}

func (b *blockState) feed(d streamDelta) {
	s := b.streams[d.stream]
	if s == nil {
		s = newDeltaStreamState()
		b.streams[d.stream] = s
	}
	s.feed(d)
}

func (b *blockState) clone() *blockState {
	c := newBlockState(b.typ)
	for name, s := range b.streams {
		c.streams[name] = s.clone()
	}
	return c
}

// digest folds the block's type and every delta stream into one value, in a
// canonical (name-sorted) order so both sides of a resume compute the same
// thing.
func (b *blockState) digest() uint64 {
	h := uint64(fnvOffset64)
	h = fnv1aString(h, b.typ)
	for _, name := range slices.Sorted(maps.Keys(b.streams)) {
		s := b.streams[name]
		h = fnv1aString(h, name)
		h = fnv1aUint(h, s.length)
		h = fnv1aUint(h, s.hash)
	}
	return h
}

// foldCompleted folds a completed block into the rolling digest of completed
// blocks: the chain value is itself the checkpoint state.
func foldCompleted(completed uint64, index int64, b *blockState) uint64 {
	h := fnv1aUint(completed, uint64(index))
	return fnv1aUint(h, b.digest())
}

// deliveredState is the engine's source of truth for everything delivered so
// far. The checkpoint, the partial snapshot and the usage accounting all
// derive from it, so they can never disagree.
type deliveredState struct {
	events          uint64
	bytes           uint64
	resumed         bool
	msgStarted      bool
	msgDeltas       uint64
	msgStopped      bool
	completedBlocks uint64
	completedDigest uint64
	open            map[int64]*blockState
}

func newDeliveredState() deliveredState {
	return deliveredState{
		completedDigest: fnvOffset64,
		open:            map[int64]*blockState{},
	}
}

func (d *deliveredState) totalBlocks() uint64 {
	return d.completedBlocks + uint64(len(d.open))
}

// replayBlock tracks how far one replayed content block has caught up with
// the delivered prefix during the current attempt.
type replayBlock struct {
	blockState
	// target is the delivered state of the block when the attempt began, if
	// the block was still open then; nil for blocks already completed, whose
	// content is verified through the completed-block digest instead.
	target  *blockState
	stopped bool
}

func (rb *replayBlock) caughtUp() bool {
	for name, ts := range rb.target.streams {
		rs := rb.streams[name]
		if rs == nil || rs.length < ts.length {
			return false
		}
	}
	return true
}

// attemptBase is the delivered state frozen at the start of an attempt: the
// target the replayed prefix is checked against. It never mutates during the
// attempt, so content delivered during the attempt cannot move the goalposts.
type attemptBase struct {
	msgStarted      bool
	msgDeltas       uint64
	msgStopped      bool
	completedBlocks uint64
	completedDigest uint64
	totalBlocks     uint64
	open            map[int64]*blockState
}

func (d *deliveredState) freeze() *attemptBase {
	open := make(map[int64]*blockState, len(d.open))
	for i, b := range d.open {
		open[i] = b.clone()
	}
	return &attemptBase{
		msgStarted:      d.msgStarted,
		msgDeltas:       d.msgDeltas,
		msgStopped:      d.msgStopped,
		completedBlocks: d.completedBlocks,
		completedDigest: d.completedDigest,
		totalBlocks:     d.totalBlocks(),
		open:            open,
	}
}

// attemptState tracks one connection attempt's progress through the replayed
// prefix.
type attemptState struct {
	base            *attemptBase
	msgStarted      bool
	msgDeltas       uint64
	completed       uint64
	completedDigest uint64
	blocks          map[int64]*replayBlock
}

// resumeEngineOps adapts the generic engine to one Messages API flavor
// (stable or beta) without re-implementing decode or accumulation.
type resumeEngineOps[T any, M any] struct {
	// accumulate is the API's existing incremental accumulator.
	accumulate func(*M, T) error
	// info extracts what the engine needs from one decoded event.
	info func(T) eventInfo
	// trim returns the event with its delta reduced to the part after the
	// delivered prefix, for a delta that straddles the resume boundary.
	trim func(event T, stream, rest string) T
	// snapshot clones the accumulated message and lists the blocks whose
	// tool input is not yet complete.
	snapshot func(*M, map[int64]struct{}) (M, []int64)
}

// resumableStream is the engine behind [ResumableMessageStream] and
// [BetaResumableMessageStream]. It is not safe for concurrent use, like
// [ssestream.Stream].
type resumableStream[T any, M any] struct {
	ctx         context.Context
	cfg         ResumableStreamConfig
	ops         resumeEngineOps[T, M]
	connect     func(context.Context) *ssestream.Stream[T]
	requestHash uint64

	acc *M
	// accEnabled is false when the stream was restored from a checkpoint
	// without an InitialSnapshot: the content that predates the checkpoint is
	// gone, so the partial result cannot be accumulated and events are only
	// delivered.
	accEnabled bool
	delivered  deliveredState

	stream  *ssestream.Stream[T]
	attempt *attemptState

	stats    ResumeStats
	cur      T
	err      error
	done     bool
	complete bool
	closed   bool
}

func newResumableStream[T any, M any](
	ctx context.Context,
	cfg *ResumableStreamConfig,
	ops resumeEngineOps[T, M],
	connect func(context.Context) *ssestream.Stream[T],
	requestHash uint64,
) *resumableStream[T, M] {
	c := cfg.normalized()
	s := &resumableStream[T, M]{
		ctx:         ctx,
		cfg:         c,
		ops:         ops,
		connect:     connect,
		requestHash: requestHash,
		acc:         new(M),
		accEnabled:  true,
		delivered:   newDeliveredState(),
	}
	if c.InitialSnapshot != nil {
		switch snap := c.InitialSnapshot.(type) {
		case M:
			*s.acc = snap
		case *M:
			if snap != nil {
				*s.acc = *snap
			}
		default:
			s.err = &ResumeImpossibleError{
				Reason:     "InitialSnapshot does not hold a message of this API",
				Checkpoint: c.Checkpoint,
			}
			s.done = true
			return s
		}
	}
	cp := c.Checkpoint
	if cp.IsZero() {
		return s
	}
	if err := cp.validate(); err != nil {
		s.err, s.done = err, true
		return s
	}
	if cp.RequestHash != 0 && requestHash != 0 && cp.RequestHash != requestHash {
		s.err = &ResumeImpossibleError{
			Reason:     "the checkpoint belongs to a different request",
			Checkpoint: cp,
		}
		s.done = true
		return s
	}
	s.delivered = deliveredState{
		events:          cp.Events,
		bytes:           cp.Bytes,
		resumed:         true,
		msgStarted:      cp.MessageStarted,
		msgDeltas:       cp.MessageDeltas,
		msgStopped:      cp.MessageStopped,
		completedBlocks: cp.CompletedBlocks,
		completedDigest: cp.CompletedDigest,
		open:            map[int64]*blockState{},
	}
	if s.delivered.completedDigest == 0 {
		s.delivered.completedDigest = fnvOffset64
	}
	for _, b := range cp.OpenBlocks {
		bs := newBlockState(b.Type)
		for _, mark := range b.Streams {
			bs.streams[mark.Name] = &deltaStreamState{length: mark.Len, hash: mark.Hash}
		}
		s.delivered.open[b.Index] = bs
	}
	if s.delivered.msgStopped {
		// The stream already terminated; there is nothing to reconnect for.
		s.complete = true
	}
	if c.InitialSnapshot == nil {
		// The delivered content predating the checkpoint is not available to
		// this stream, so its partial result cannot be accumulated.
		s.accEnabled = false
	}
	return s
}

// Next advances the stream, reconnecting and splicing after interruptions.
// The delivered event sequence is one continuous stream: message_start and
// message_stop appear exactly once, content blocks keep their order, and
// replayed content is delivered once.
func (s *resumableStream[T, M]) Next() bool {
	if s.done || s.closed {
		return false
	}
	if s.complete {
		s.finish()
		return false
	}
	for {
		if s.stream == nil {
			s.beginAttempt()
		}
		if s.stream.Next() {
			event := s.stream.Current()
			out, deliver, err := s.process(event)
			if err != nil {
				s.fail(err)
				return false
			}
			if !deliver {
				continue
			}
			if err := s.deliver(out); err != nil {
				s.fail(err)
				return false
			}
			s.cur = out
			if s.delivered.msgStopped {
				// message_stop is delivered exactly once; nothing after it is
				// part of the stream, including repeated stop notifications.
				s.complete = true
			}
			return true
		}

		cause := s.stream.Err()
		s.endAttempt()
		if s.delivered.msgStopped {
			// The message was complete before the connection ended; whatever
			// the connection did afterwards is not an interruption of it.
			s.done = true
			return false
		}
		inter := s.classify(cause)
		s.stats.Interruptions++
		if !inter.resumable || s.cfg.MaxResumes == 0 {
			s.err = s.terminalError(inter, cause)
			s.done = true
			return false
		}
		if s.stats.Resumes >= uint64(s.cfg.MaxResumes) {
			s.err = &ResumeBudgetExhaustedError{
				Checkpoint: s.Checkpoint(),
				Stats:      s.Stats(),
				Err:        cause,
			}
			s.done = true
			return false
		}
		if !s.waitBackoff(inter) {
			s.err = s.terminalError(s.classify(s.ctx.Err()), s.ctx.Err())
			s.done = true
			return false
		}
		s.stats.Resumes++
	}
}

// Current returns the event most recently delivered by Next.
func (s *resumableStream[T, M]) Current() T {
	return s.cur
}

// Err returns the error that ended the stream, if any. Interruptions that
// could not be resumed are reported as [*StreamInterruptError],
// [*ResumeBudgetExhaustedError], [*ResumeInconsistentError] or
// [*ResumeImpossibleError]; a non-retryable API error is returned as the
// original [*apierror.Error].
func (s *resumableStream[T, M]) Err() error {
	return s.err
}

// Close releases the current connection. A closed stream is never resumed.
func (s *resumableStream[T, M]) Close() error {
	s.closed = true
	if s.stream != nil {
		return s.stream.Close()
	}
	return nil
}

// All returns an iterator over the remaining events of the stream.
func (s *resumableStream[T, M]) All() iter.Seq[T] {
	return func(yield func(T) bool) {
		for s.Next() {
			if !yield(s.Current()) {
				break
			}
		}
	}
}

// Checkpoint returns the position the stream has delivered up to. It is safe
// to call between events; the returned value is a snapshot.
func (s *resumableStream[T, M]) Checkpoint() StreamCheckpoint {
	d := &s.delivered
	c := StreamCheckpoint{
		Version:         streamCheckpointVersion,
		RequestHash:     s.requestHash,
		Events:          d.events,
		Bytes:           d.bytes,
		Resumed:         d.resumed,
		MessageStarted:  d.msgStarted,
		MessageDeltas:   d.msgDeltas,
		MessageStopped:  d.msgStopped,
		CompletedBlocks: d.completedBlocks,
		CompletedDigest: d.completedDigest,
	}
	for index, b := range d.open {
		bc := BlockCheckpoint{Index: index, Type: b.typ}
		for _, name := range slices.Sorted(maps.Keys(b.streams)) {
			st := b.streams[name]
			bc.Streams = append(bc.Streams, DeltaStreamMark{Name: name, Len: st.length, Hash: st.hash})
		}
		c.OpenBlocks = append(c.OpenBlocks, bc)
	}
	slices.SortFunc(c.OpenBlocks, func(a, b BlockCheckpoint) int {
		switch {
		case a.Index < b.Index:
			return -1
		case a.Index > b.Index:
			return 1
		}
		return 0
	})
	return c
}

// Stats returns the stream's counters. Event and byte counts are cumulative
// across checkpoint restores; interruption, resume and dedup counters cover
// this stream's lifetime.
func (s *resumableStream[T, M]) Stats() ResumeStats {
	stats := s.stats
	stats.EventsDelivered = s.delivered.events
	stats.BytesDelivered = s.delivered.bytes
	return stats
}

// streamSnapshot is a partial result of a resumable stream at one position.
type streamSnapshot[M any] struct {
	// Message is the message accumulated so far. Blocks listed in OpenBlocks
	// are still growing. It is meaningful only when MessageAvailable is true.
	Message M
	// MessageAvailable is false when the stream was restored from a
	// checkpoint without an InitialSnapshot: the content delivered before the
	// checkpoint belongs to the previous stream, so this stream cannot offer
	// it.
	MessageAvailable bool
	// Complete is true once message_stop was delivered.
	Complete bool
	// Resumed is true when the message was assembled across more than one
	// attempt, so the caller can tell a spliced result from a single-run one.
	Resumed bool
	// OpenBlocks lists the content blocks whose stop event has not been
	// delivered; they stopped mid-block.
	OpenBlocks []int64
	// IncompleteToolInputs lists the tool-use blocks whose input has not
	// closed. Their Input is partial and must not be executed.
	IncompleteToolInputs []int64
}

// Snapshot returns the partial result corresponding to the current
// checkpoint: which blocks are complete and which stopped mid-block, with
// unclosed tool inputs marked incomplete.
func (s *resumableStream[T, M]) Snapshot() streamSnapshot[M] {
	open := make(map[int64]struct{}, len(s.delivered.open))
	blocks := make([]int64, 0, len(s.delivered.open))
	for index := range s.delivered.open {
		open[index] = struct{}{}
		blocks = append(blocks, index)
	}
	slices.Sort(blocks)
	message, incomplete := s.ops.snapshot(s.acc, open)
	return streamSnapshot[M]{
		Message:              message,
		MessageAvailable:     s.accEnabled,
		Complete:             s.delivered.msgStopped,
		Resumed:              s.delivered.resumed || s.stats.Resumes > 0,
		OpenBlocks:           blocks,
		IncompleteToolInputs: incomplete,
	}
}

// beginAttempt opens one connection attempt.
func (s *resumableStream[T, M]) beginAttempt() {
	s.attempt = &attemptState{
		base:            s.delivered.freeze(),
		completedDigest: fnvOffset64,
		blocks:          map[int64]*replayBlock{},
	}
	s.stream = s.connect(s.ctx)
}

// endAttempt releases the interrupted attempt's connection, leaving no
// half-open state behind; the delivered position does not move.
func (s *resumableStream[T, M]) endAttempt() {
	if s.stream != nil {
		s.stream.Close()
		s.stream = nil
	}
	s.attempt = nil
}

func (s *resumableStream[T, M]) finish() {
	s.endAttempt()
	s.done = true
}

func (s *resumableStream[T, M]) fail(err error) {
	s.err = err
	s.finish()
}

// process classifies one event of the current attempt against the delivered
// prefix. It returns the event to deliver (possibly trimmed) or, when the
// event is a duplicate of delivered content, deliver == false.
func (s *resumableStream[T, M]) process(event T) (out T, deliver bool, err error) {
	info := s.ops.info(event)
	a := s.attempt
	base := a.base
	switch info.typ {
	case "message_start":
		if !base.msgStarted {
			return event, true, nil
		}
		if a.msgStarted {
			return event, false, s.inconsistentf("message_start arrived twice in one attempt")
		}
		a.msgStarted = true
		s.dropDuplicate(info)
		return event, false, nil

	case "content_block_start":
		if info.index >= int64(base.totalBlocks) {
			return event, true, nil
		}
		if _, dup := a.blocks[info.index]; dup {
			return event, false, s.inconsistentf("content block %d started twice in one attempt", info.index)
		}
		rb := &replayBlock{
			blockState: blockState{typ: info.blockType, streams: map[string]*deltaStreamState{}},
			target:     base.open[info.index],
		}
		if rb.target != nil && rb.target.typ != info.blockType {
			return event, false, s.inconsistentf("content block %d changed type between attempts", info.index)
		}
		a.blocks[info.index] = rb
		s.dropDuplicate(info)
		return event, false, nil

	case "content_block_delta":
		if info.index >= int64(base.totalBlocks) {
			return event, true, nil
		}
		rb := a.blocks[info.index]
		if rb == nil {
			return event, false, s.inconsistentf("content block %d sent a delta before its start", info.index)
		}
		if rb.stopped {
			return event, false, s.inconsistentf("content block %d sent a delta after its stop", info.index)
		}
		if rb.target == nil {
			// A block already completed in the delivered prefix: the whole
			// delta is a duplicate, verified later through the completed
			// digest.
			if info.delta != nil {
				rb.feed(*info.delta)
			}
			s.dropDuplicate(info)
			return event, false, nil
		}
		return s.catchUp(event, info, rb)

	case "content_block_stop":
		if info.index >= int64(base.totalBlocks) {
			return event, true, nil
		}
		rb := a.blocks[info.index]
		if rb == nil {
			return event, false, s.inconsistentf("content block %d stopped before its start", info.index)
		}
		if rb.stopped {
			return event, false, s.inconsistentf("content block %d stopped twice", info.index)
		}
		rb.stopped = true
		if rb.target == nil {
			a.completed++
			a.completedDigest = foldCompleted(a.completedDigest, info.index, &rb.blockState)
			if a.completed > base.completedBlocks {
				return event, false, s.inconsistentf("more content blocks completed than were delivered")
			}
			if a.completed == base.completedBlocks && a.completedDigest != base.completedDigest {
				return event, false, s.inconsistentf("the completed content blocks differ from what was delivered")
			}
			s.dropDuplicate(info)
			return event, false, nil
		}
		if !rb.caughtUp() {
			return event, false, s.inconsistentf("content block %d ended before reaching the delivered content", info.index)
		}
		// The delivered stream never saw this stop: it is new.
		return event, true, nil

	case "message_delta":
		if a.msgDeltas < base.msgDeltas {
			a.msgDeltas++
			s.dropDuplicate(info)
			return event, false, nil
		}
		return event, true, nil

	case "message_stop":
		if base.msgStopped {
			s.dropDuplicate(info)
			return event, false, nil
		}
		if err := s.checkCaughtUp(); err != nil {
			return event, false, err
		}
		return event, true, nil
	}
	// Events of a type this SDK does not model are passed through.
	return event, true, nil
}

// catchUp handles a content_block_delta for a block that was still open when
// the attempt began: the part overlapping the delivered prefix is verified
// and dropped, the rest is delivered.
func (s *resumableStream[T, M]) catchUp(event T, info eventInfo, rb *replayBlock) (T, bool, error) {
	if info.delta == nil {
		s.dropDuplicate(info)
		return event, false, nil
	}
	d := *info.delta
	ts := rb.target.streams[d.stream]
	if ts == nil || ts.length == 0 {
		// The block had nothing of this stream yet: the delta is new.
		return event, true, nil
	}
	rs := rb.streams[d.stream]
	if rs == nil {
		rs = newDeltaStreamState()
		rb.streams[d.stream] = rs
	}
	if rs.length >= ts.length {
		// The delivered prefix of this stream is fully replayed.
		return event, true, nil
	}
	if d.kind == deltaItems {
		rs.feed(d)
		if rs.length == ts.length && rs.hash != ts.hash {
			return event, false, s.inconsistentf("content block %d stream %q differs from what was delivered", info.index, d.stream)
		}
		s.dropDuplicate(info)
		return event, false, nil
	}
	payload := uint64(len(d.payload))
	switch {
	case rs.length+payload < ts.length:
		rs.feed(d)
		s.dropDuplicate(info)
		return event, false, nil
	case rs.length+payload == ts.length:
		rs.feed(d)
		if rs.hash != ts.hash {
			return event, false, s.inconsistentf("content block %d stream %q differs from what was delivered", info.index, d.stream)
		}
		s.dropDuplicate(info)
		return event, false, nil
	default:
		// The delta straddles the resume boundary: verify the overlapping
		// head, deliver the rest.
		keep := ts.length - rs.length
		rs.feedBytes(deltaBytes, d.payload[:keep])
		if rs.hash != ts.hash {
			return event, false, s.inconsistentf("content block %d stream %q differs from what was delivered", info.index, d.stream)
		}
		s.stats.DedupTrimmedEvents++
		s.stats.DedupDroppedBytes += keep
		return s.ops.trim(event, d.stream, d.payload[keep:]), true, nil
	}
}

// checkCaughtUp verifies at message_stop that the replay covered the whole
// delivered prefix.
func (s *resumableStream[T, M]) checkCaughtUp() error {
	a := s.attempt
	base := a.base
	if a.msgDeltas < base.msgDeltas {
		return s.inconsistentf("message_delta events delivered before are missing from the resumed stream")
	}
	if a.completed != base.completedBlocks {
		return s.inconsistentf("completed content blocks delivered before are missing from the resumed stream")
	}
	for index, target := range base.open {
		rb := a.blocks[index]
		if rb == nil {
			return s.inconsistentf("content block %d delivered before is missing from the resumed stream", index)
		}
		rb.target = target
		if !rb.caughtUp() {
			return s.inconsistentf("content block %d is missing delivered content in the resumed stream", index)
		}
	}
	return nil
}

// deliver hands one event to the application: it advances the accumulated
// message, the checkpoint state and the stats together, so the position never
// moves ahead of delivered content.
func (s *resumableStream[T, M]) deliver(event T) error {
	info := s.ops.info(event)
	if s.accEnabled {
		if err := s.ops.accumulate(s.acc, event); err != nil {
			return fmt.Errorf("anthropic: cannot accumulate streamed event: %w", err)
		}
	}
	d := &s.delivered
	d.events++
	d.bytes += uint64(len(info.raw))
	switch info.typ {
	case "message_start":
		d.msgStarted = true
	case "message_delta":
		d.msgDeltas++
	case "message_stop":
		d.msgStopped = true
	case "content_block_start":
		d.open[info.index] = newBlockState(info.blockType)
	case "content_block_delta":
		if b := d.open[info.index]; b != nil && info.delta != nil {
			b.feed(*info.delta)
		}
	case "content_block_stop":
		if b, ok := d.open[info.index]; ok {
			d.completedDigest = foldCompleted(d.completedDigest, info.index, b)
			d.completedBlocks++
			delete(d.open, info.index)
		}
	}
	return nil
}

func (s *resumableStream[T, M]) dropDuplicate(info eventInfo) {
	s.stats.DedupDroppedEvents++
	s.stats.DedupDroppedBytes += uint64(len(info.raw))
}

func (s *resumableStream[T, M]) inconsistentf(format string, args ...any) error {
	return &ResumeInconsistentError{
		Checkpoint: s.Checkpoint(),
		Detail:     fmt.Sprintf(format, args...),
	}
}

// streamInterruption is the classification of one attempt's end.
type streamInterruption struct {
	kind       InterruptionKind
	resumable  bool
	retryAfter time.Duration
}

func (s *resumableStream[T, M]) classify(err error) streamInterruption {
	if err == nil {
		// A clean end of the response before message_stop.
		return streamInterruption{kind: InterruptionTruncated, resumable: true}
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		if ctxErr := s.ctx.Err(); ctxErr != nil {
			// The caller's own context ended the stream; never resume it.
			if errors.Is(ctxErr, context.DeadlineExceeded) {
				return streamInterruption{kind: InterruptionCallerTimeout}
			}
			return streamInterruption{kind: InterruptionCallerCancel}
		}
		if errors.Is(err, context.DeadlineExceeded) {
			// A per-attempt timeout fired; the caller is still waiting.
			return streamInterruption{kind: InterruptionAttemptTimeout, resumable: true}
		}
		return streamInterruption{kind: InterruptionConnectionReset, resumable: true}
	}
	var aerr *apierror.Error
	if errors.As(err, &aerr) {
		return streamInterruption{
			kind:       InterruptionUpstreamError,
			resumable:  resumableAPIError(aerr),
			retryAfter: retryAfterOf(aerr),
		}
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return streamInterruption{kind: InterruptionTruncated, resumable: true}
	}
	var nerr net.Error
	if errors.As(err, &nerr) {
		return streamInterruption{kind: InterruptionConnectionReset, resumable: true}
	}
	if errors.Is(err, io.EOF) {
		return streamInterruption{kind: InterruptionTruncated, resumable: true}
	}
	return streamInterruption{kind: InterruptionUnknown}
}

// resumableAPIError reports whether an API error allows resuming, following
// the same signals as the request-level retry logic: the x-should-retry
// header first, then the status class, then the error type.
func resumableAPIError(aerr *apierror.Error) bool {
	if res := aerr.Response; res != nil {
		switch res.Header.Get("x-should-retry") {
		case "true":
			return true
		case "false":
			return false
		}
	}
	switch aerr.StatusCode {
	case 408, 409, 429:
		return true
	}
	if aerr.StatusCode >= 500 {
		return true
	}
	switch aerr.Type() {
	case shared.ErrorTypeOverloadedError, shared.ErrorTypeAPIError,
		shared.ErrorTypeRateLimitError, shared.ErrorTypeTimeoutError:
		return true
	}
	return false
}

// retryAfterOf reads the server's retry hint, like the request-level retry
// logic does.
func retryAfterOf(aerr *apierror.Error) time.Duration {
	res := aerr.Response
	if res == nil {
		return 0
	}
	if v := res.Header.Get("Retry-After-Ms"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return time.Duration(f * float64(time.Millisecond))
		}
	}
	if v := res.Header.Get("Retry-After"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return time.Duration(f * float64(time.Second))
		}
		if t, err := http.ParseTime(v); err == nil {
			return time.Until(t)
		}
	}
	return 0
}

// terminalError builds the error a non-resumable interruption ends the stream
// with. API errors are propagated as they are.
func (s *resumableStream[T, M]) terminalError(inter streamInterruption, cause error) error {
	var aerr *apierror.Error
	if errors.As(cause, &aerr) {
		return aerr
	}
	return &StreamInterruptError{
		Kind:       inter.kind,
		Checkpoint: s.Checkpoint(),
		Err:        cause,
	}
}

// waitBackoff waits between attempts, honoring the server's retry hint, and
// reports whether the wait completed. The timer is stopped when the caller's
// context ends first.
func (s *resumableStream[T, M]) waitBackoff(inter streamInterruption) bool {
	delay := s.cfg.InitialWait
	for i := uint64(0); i < s.stats.Resumes && delay < s.cfg.MaxWait; i++ {
		delay *= 2
	}
	if delay > s.cfg.MaxWait {
		delay = s.cfg.MaxWait
	}
	if inter.retryAfter > 0 {
		delay = inter.retryAfter
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-s.ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// requestHashOf identifies the request a checkpoint belongs to, as a hash of
// the serialized parameters. The parameters themselves are never stored.
func requestHashOf(params any) uint64 {
	data, err := json.Marshal(params)
	if err != nil {
		return 0
	}
	return fnv1a(fnvOffset64, data)
}

// deltaStreamField maps a byte-stream name to the JSON field of the delta
// that carries its payload.
var deltaStreamField = map[string]string{
	"text":       "text",
	"input_json": "partial_json",
	"thinking":   "thinking",
	"signature":  "signature",
}

// trimDeltaRaw rewrites the raw JSON of an event whose delta was trimmed to
// the part after the resume boundary.
func trimDeltaRaw(setRaw func(path, value string), stream, rest string) {
	field, ok := deltaStreamField[stream]
	if !ok {
		return
	}
	setRaw(field, jsonString(rest))
}

// stableStreamEventInfo adapts [MessageStreamEventUnion] to the engine.
func stableStreamEventInfo(event MessageStreamEventUnion) eventInfo {
	info := eventInfo{typ: event.Type, index: event.Index, raw: event.RawJSON()}
	switch event.Type {
	case "content_block_start":
		info.blockType = gjson.Get(event.ContentBlock.RawJSON(), "type").String()
	case "content_block_delta":
		info.delta = stableStreamDelta(event.Delta, gjson.Get(info.raw, "delta").Raw)
	}
	return info
}

func stableStreamDelta(d MessageStreamEventUnionDelta, deltaRaw string) *streamDelta {
	switch d.Type {
	case "text_delta":
		return &streamDelta{stream: "text", kind: deltaBytes, payload: d.Text}
	case "input_json_delta":
		return &streamDelta{stream: "input_json", kind: deltaBytes, payload: d.PartialJSON}
	case "thinking_delta":
		return &streamDelta{stream: "thinking", kind: deltaBytes, payload: d.Thinking}
	case "signature_delta":
		return &streamDelta{stream: "signature", kind: deltaBytes, payload: d.Signature}
	case "citations_delta":
		return &streamDelta{stream: "citations", kind: deltaItems, payload: d.Citation.RawJSON()}
	case "":
		return nil
	default:
		return &streamDelta{stream: "other:" + d.Type, kind: deltaItems, payload: deltaRaw}
	}
}

func trimStableStreamEvent(event MessageStreamEventUnion, stream, rest string) MessageStreamEventUnion {
	switch stream {
	case "text":
		event.Delta.Text = rest
	case "input_json":
		event.Delta.PartialJSON = rest
	case "thinking":
		event.Delta.Thinking = rest
	case "signature":
		event.Delta.Signature = rest
	}
	trimDeltaRaw(func(field, value string) {
		event.JSON.raw, _ = sjson.SetRaw(event.JSON.raw, "delta."+field, value)
		event.Delta.JSON.raw, _ = sjson.SetRaw(event.Delta.JSON.raw, field, value)
	}, stream, rest)
	return event
}

func snapshotMessage(acc *Message, open map[int64]struct{}) (Message, []int64) {
	clone := *acc
	clone.Content = slices.Clone(acc.Content)
	for i := range clone.Content {
		clone.Content[i].Input = slices.Clone(clone.Content[i].Input)
		clone.Content[i].Citations = slices.Clone(clone.Content[i].Citations)
	}
	var incomplete []int64
	for i := range clone.Content {
		if _, ok := open[int64(i)]; ok && isToolInputBlock(clone.Content[i].Type) {
			incomplete = append(incomplete, int64(i))
		}
	}
	return clone, incomplete
}

// isToolInputBlock reports whether blocks of this type carry a tool input
// that streams in as partial JSON.
func isToolInputBlock(blockType string) bool {
	switch blockType {
	case "tool_use", "server_tool_use", "mcp_tool_use":
		return true
	}
	return false
}

// betaStreamEventInfo adapts [BetaRawMessageStreamEventUnion] to the engine.
func betaStreamEventInfo(event BetaRawMessageStreamEventUnion) eventInfo {
	info := eventInfo{typ: event.Type, index: event.Index, raw: event.RawJSON()}
	switch event.Type {
	case "content_block_start":
		info.blockType = gjson.Get(event.ContentBlock.RawJSON(), "type").String()
	case "content_block_delta":
		info.delta = betaStreamDelta(event.Delta, gjson.Get(info.raw, "delta").Raw)
	}
	return info
}

func betaStreamDelta(d BetaRawMessageStreamEventUnionDelta, deltaRaw string) *streamDelta {
	switch d.Type {
	case "text_delta":
		return &streamDelta{stream: "text", kind: deltaBytes, payload: d.Text}
	case "input_json_delta":
		return &streamDelta{stream: "input_json", kind: deltaBytes, payload: d.PartialJSON}
	case "thinking_delta":
		return &streamDelta{stream: "thinking", kind: deltaBytes, payload: d.Thinking}
	case "signature_delta":
		return &streamDelta{stream: "signature", kind: deltaBytes, payload: d.Signature}
	case "citations_delta":
		return &streamDelta{stream: "citations", kind: deltaItems, payload: d.Citation.RawJSON()}
	case "":
		return nil
	default:
		return &streamDelta{stream: "other:" + d.Type, kind: deltaItems, payload: deltaRaw}
	}
}

func trimBetaStreamEvent(event BetaRawMessageStreamEventUnion, stream, rest string) BetaRawMessageStreamEventUnion {
	switch stream {
	case "text":
		event.Delta.Text = rest
	case "input_json":
		event.Delta.PartialJSON = rest
	case "thinking":
		event.Delta.Thinking = rest
	case "signature":
		event.Delta.Signature = rest
	}
	trimDeltaRaw(func(field, value string) {
		event.JSON.raw, _ = sjson.SetRaw(event.JSON.raw, "delta."+field, value)
		event.Delta.JSON.raw, _ = sjson.SetRaw(event.Delta.JSON.raw, field, value)
	}, stream, rest)
	return event
}

func snapshotBetaMessage(acc *BetaMessage, open map[int64]struct{}) (BetaMessage, []int64) {
	clone := *acc
	clone.Content = slices.Clone(acc.Content)
	for i := range clone.Content {
		clone.Content[i].Input = slices.Clone(clone.Content[i].Input)
		clone.Content[i].Citations = slices.Clone(clone.Content[i].Citations)
	}
	var incomplete []int64
	for i := range clone.Content {
		if _, ok := open[int64(i)]; ok && isToolInputBlock(clone.Content[i].Type) {
			incomplete = append(incomplete, int64(i))
		}
	}
	return clone, incomplete
}

// ResumableMessageStream is the streaming counterpart of
// [ssestream.Stream] with resumption, returned by
// [MessageService.NewResumableStreaming].
type ResumableMessageStream = resumableStream[MessageStreamEventUnion, Message]

// BetaResumableMessageStream is [ResumableMessageStream] for the beta
// Messages API.
type BetaResumableMessageStream = resumableStream[BetaRawMessageStreamEventUnion, BetaMessage]

// MessageStreamSnapshot is the partial result of a [ResumableMessageStream].
type MessageStreamSnapshot = streamSnapshot[Message]

// BetaMessageStreamSnapshot is the partial result of a
// [BetaResumableMessageStream].
type BetaMessageStreamSnapshot = streamSnapshot[BetaMessage]

// NewResumableStreaming starts a streaming message creation that resumes
// itself after interruptions: when the connection drops before message_stop,
// the stream re-issues the same request, verifies the replayed prefix against
// what was already delivered, drops the duplicates and delivers only the
// continuation. cfg may be nil for defaults; its Checkpoint continues a
// stream from a previously captured position, possibly in another process.
//
// The method only adds behavior; [MessageService.NewStreaming] is unchanged.
func (r *MessageService) NewResumableStreaming(ctx context.Context, params MessageNewParams, cfg *ResumableStreamConfig, opts ...option.RequestOption) *ResumableMessageStream {
	return newResumableStream(ctx, cfg,
		resumeEngineOps[MessageStreamEventUnion, Message]{
			accumulate: func(acc *Message, event MessageStreamEventUnion) error {
				return acc.Accumulate(event)
			},
			info:     stableStreamEventInfo,
			trim:     trimStableStreamEvent,
			snapshot: snapshotMessage,
		},
		func(ctx context.Context) *ssestream.Stream[MessageStreamEventUnion] {
			return r.NewStreaming(ctx, params, opts...)
		},
		requestHashOf(params),
	)
}

// NewResumableStreaming is [MessageService.NewResumableStreaming] for the
// beta Messages API.
func (r *BetaMessageService) NewResumableStreaming(ctx context.Context, params BetaMessageNewParams, cfg *ResumableStreamConfig, opts ...option.RequestOption) *BetaResumableMessageStream {
	return newResumableStream(ctx, cfg,
		resumeEngineOps[BetaRawMessageStreamEventUnion, BetaMessage]{
			accumulate: func(acc *BetaMessage, event BetaRawMessageStreamEventUnion) error {
				return acc.Accumulate(event)
			},
			info:     betaStreamEventInfo,
			trim:     trimBetaStreamEvent,
			snapshot: snapshotBetaMessage,
		},
		func(ctx context.Context) *ssestream.Stream[BetaRawMessageStreamEventUnion] {
			return r.NewStreaming(ctx, params, opts...)
		},
		requestHashOf(params),
	)
}
