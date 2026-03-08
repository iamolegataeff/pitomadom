package main

import (
	"math"
	"testing"
)

// ============================================================================
// HEBREW TESTS
// ============================================================================

func TestIsHebrew(t *testing.T) {
	tests := []struct {
		ch   rune
		want bool
	}{
		{'א', true}, {'ת', true}, {'ך', true}, {'ם', true},
		{'a', false}, {'z', false}, {'1', false}, {' ', false},
	}
	for _, tt := range tests {
		got := isHebrew(tt.ch)
		if got != tt.want {
			t.Errorf("isHebrew(%c) = %v, want %v", tt.ch, got, tt.want)
		}
	}
}

func TestNormalizeLetter(t *testing.T) {
	tests := []struct {
		ch, want rune
	}{
		{'ך', 'כ'}, {'ם', 'מ'}, {'ן', 'נ'}, {'ף', 'פ'}, {'ץ', 'צ'},
		{'א', 'א'}, {'ב', 'ב'}, // regular letters unchanged
	}
	for _, tt := range tests {
		got := normalizeLetter(tt.ch)
		if got != tt.want {
			t.Errorf("normalizeLetter(%c) = %c, want %c", tt.ch, got, tt.want)
		}
	}
}

func TestCharToIdx(t *testing.T) {
	// א=0, ב=1, ..., ת=21
	if got := charToIdx('א'); got != 0 {
		t.Errorf("charToIdx(א) = %d, want 0", got)
	}
	if got := charToIdx('ת'); got != 21 {
		t.Errorf("charToIdx(ת) = %d, want 21", got)
	}
	// Finals map to regular
	if got := charToIdx('ך'); got != charToIdx('כ') {
		t.Errorf("charToIdx(ך) should equal charToIdx(כ)")
	}
	// Unknown
	if got := charToIdx('x'); got != unkLetter {
		t.Errorf("charToIdx(x) = %d, want %d", got, unkLetter)
	}
}

func TestExtractHebrewWords(t *testing.T) {
	tests := []struct {
		text string
		want int // number of words
	}{
		{"שלום עולם", 2},
		{"אהבה וחסד ושלום", 3},
		{"hello world", 0},
		{"שלום hello עולם", 2},
		{"", 0},
	}
	for _, tt := range tests {
		got := extractHebrewWords(tt.text)
		if len(got) != tt.want {
			t.Errorf("extractHebrewWords(%q) = %d words, want %d", tt.text, len(got), tt.want)
		}
	}
}

func TestExtractRoot(t *testing.T) {
	// Must match Python train_rtl.py extract_root exactly
	// These are the expected (c1, c2, c3) indices from Python
	tests := []struct {
		word       string
		c1, c2, c3 int
		ok         bool
	}{
		// חסד: 3 consonants, no stripping -> ח(7).ס(14).ד(3)
		{"חסד", 7, 14, 3, true},
		// עולם: ע.ו.ל.מ -> strip nothing with >=2 remaining after each
		// prefix ו matches? No, עולמ starts with ע not ו
		// Actually: consonants are ע.ו.ל.מ (4), > 3, take first 3: ע(15).ו(5).ל(11)
		{"עולם", 15, 5, 11, true},
		// שלום: consonants ש.ל.ו.מ -> strip prefix של (matches, 4-2=2>=2) -> ו.מ -> pad -> ו(5).מ(12).מ(12)
		{"שלום", 5, 12, 12, true},
		// אהבה: consonants א.ה.ב.ה -> strip prefix א (4-1=3>=2) -> ה.ב.ה -> ה(4).ב(1).ה(4)
		// Wait, Python gives (4, 1, 1) = ה.ב.ב. Let me check suffix stripping.
		// After prefix א stripped: הבה (3 chars). Suffix ה matches (3-1=2>=2) -> הב -> pad -> ה(4).ב(1).ב(1)
		{"אהבה", 4, 1, 1, true},
		// ברכה: consonants ב.ר.כ.ה -> strip prefix ב (4-1=3>=2) -> ר.כ.ה (3 chars)
		// suffix ה matches (3-1=2>=2) -> ר.כ -> pad -> ר(19).כ(10).כ(10)
		{"ברכה", 19, 10, 10, true},
		// Too short
		{"א", 0, 0, 0, false},
	}
	for _, tt := range tests {
		c1, c2, c3, ok := extractRoot(tt.word)
		if ok != tt.ok {
			t.Errorf("extractRoot(%q) ok=%v, want %v", tt.word, ok, tt.ok)
			continue
		}
		if ok && (c1 != tt.c1 || c2 != tt.c2 || c3 != tt.c3) {
			t.Errorf("extractRoot(%q) = (%d,%d,%d), want (%d,%d,%d)",
				tt.word, c1, c2, c3, tt.c1, tt.c2, tt.c3)
		}
	}
}

func TestGematria(t *testing.T) {
	// Standard gematria values
	if g := heGematria['א']; g != 1 {
		t.Errorf("gematria(א) = %d, want 1", g)
	}
	if g := heGematria['ת']; g != 400 {
		t.Errorf("gematria(ת) = %d, want 400", g)
	}
	if g := heGematria['ק']; g != 100 {
		t.Errorf("gematria(ק) = %d, want 100", g)
	}
	// Finals should have same value as regular
	if heGematria['ך'] != heGematria['כ'] {
		t.Errorf("gematria(ך) != gematria(כ)")
	}
}

func TestRootGematria(t *testing.T) {
	// ח(8) + ס(60) + ד(4) = 72, normalized = 72/500 = 0.144
	g := rootGematria(7, 14, 3) // ח, ס, ד
	expected := float32(72.0 / 500.0)
	if math.Abs(float64(g-expected)) > 0.001 {
		t.Errorf("rootGematria(ח,ס,ד) = %f, want %f", g, expected)
	}
}

func TestRootToString(t *testing.T) {
	// ח.ס.ד
	got := rootToString(7, 14, 3)
	if got != "ח.ס.ד" {
		t.Errorf("rootToString(7,14,3) = %q, want %q", got, "ח.ס.ד")
	}
}

// ============================================================================
// TENSOR OP TESTS
// ============================================================================

func TestMatvec(t *testing.T) {
	// 2x3 matrix @ 3-vector
	W := []float32{1, 2, 3, 4, 5, 6}
	x := []float32{1, 0, 1}
	out := matvec(W, x, 2, 3)
	// [1*1+2*0+3*1, 4*1+5*0+6*1] = [4, 10]
	if out[0] != 4 || out[1] != 10 {
		t.Errorf("matvec = %v, want [4, 10]", out)
	}
}

func TestLayerNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	gamma := []float32{1, 1, 1, 1}
	beta := []float32{0, 0, 0, 0}
	out := layerNorm(x, gamma, beta, 4)

	// Mean=2.5, Var=1.25, std=sqrt(1.25+1e-6)
	// Should be approximately centered around 0
	sum := float32(0)
	for _, v := range out {
		sum += v
	}
	if math.Abs(float64(sum)) > 0.01 {
		t.Errorf("layerNorm output mean should be ~0, got sum=%f", sum)
	}
}

func TestSoftmax(t *testing.T) {
	x := []float32{1, 2, 3}
	softmax(x, 3)

	sum := float32(0)
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 0.001 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}
	// Largest input should have largest probability
	if x[2] <= x[1] || x[1] <= x[0] {
		t.Errorf("softmax order wrong: %v", x)
	}
}

func TestArgmax(t *testing.T) {
	tests := []struct {
		x    []float32
		want int
	}{
		{[]float32{1, 3, 2}, 1},
		{[]float32{5, 1, 2}, 0},
		{[]float32{1, 2, 5}, 2},
	}
	for _, tt := range tests {
		got := argmax(tt.x)
		if got != tt.want {
			t.Errorf("argmax(%v) = %d, want %d", tt.x, got, tt.want)
		}
	}
}

func TestGELU(t *testing.T) {
	// GELU(0) should be 0
	if g := gelu(0); math.Abs(float64(g)) > 0.001 {
		t.Errorf("gelu(0) = %f, want ~0", g)
	}
	// GELU(x) > 0 for large x
	if g := gelu(5); g <= 0 {
		t.Errorf("gelu(5) = %f, want > 0", g)
	}
	// GELU(x) ≈ x for large x
	if g := gelu(10); math.Abs(float64(g-10)) > 0.1 {
		t.Errorf("gelu(10) = %f, want ~10", g)
	}
}

func TestFloat16Conversion(t *testing.T) {
	// 0x3C00 = 1.0 in f16
	if got := float16ToFloat32(0x3C00); got != 1.0 {
		t.Errorf("f16(0x3C00) = %f, want 1.0", got)
	}
	// 0x0000 = 0.0
	if got := float16ToFloat32(0x0000); got != 0.0 {
		t.Errorf("f16(0x0000) = %f, want 0.0", got)
	}
	// 0xBC00 = -1.0
	if got := float16ToFloat32(0xBC00); got != -1.0 {
		t.Errorf("f16(0xBC00) = %f, want -1.0", got)
	}
	// 0x4000 = 2.0
	if got := float16ToFloat32(0x4000); got != 2.0 {
		t.Errorf("f16(0x4000) = %f, want 2.0", got)
	}
}

func TestRuneCount(t *testing.T) {
	if got := runeCount("שלום"); got != 4 {
		t.Errorf("runeCount(שלום) = %d, want 4", got)
	}
	if got := runeCount("abc"); got != 3 {
		t.Errorf("runeCount(abc) = %d, want 3", got)
	}
}

func TestNormalizeString(t *testing.T) {
	// Finals should be normalized
	got := normalizeString("שלום")
	want := "שלומ" // ם -> מ
	if got != want {
		t.Errorf("normalizeString(שלום) = %q, want %q", got, want)
	}
}

// ============================================================================
// INTEGRATION CONSTANTS
// ============================================================================

func TestHeLettersCount(t *testing.T) {
	if len(heLetters) != 22 {
		t.Errorf("heLetters has %d entries, want 22", len(heLetters))
	}
}

func TestGematriaComplete(t *testing.T) {
	// All 22 regular letters + 5 finals = 27 entries
	if len(heGematria) != 27 {
		t.Errorf("heGematria has %d entries, want 27", len(heGematria))
	}
}

func TestLetterIdxMapping(t *testing.T) {
	// Every letter should map to a valid index
	for _, ch := range heLetters {
		idx, ok := letterToIdx[ch]
		if !ok {
			t.Errorf("letter %c not in letterToIdx", ch)
		}
		if idx < 0 || idx >= numLetters {
			t.Errorf("letterToIdx[%c] = %d, out of range [0,%d)", ch, idx, numLetters)
		}
	}
}
