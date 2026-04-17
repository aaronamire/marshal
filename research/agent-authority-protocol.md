# Agent Authority Protocol — Design Document

Status: exploratory / future direction
Written: 2026-04-15
Author: design session, Leaves OS

## 0. Why this document exists

The internet has clean primitives for identity (PKI), transport (TCP/TLS),
data (HTTP), money (Bitcoin), programs (containers), state (databases),
and consensus (blockchains). **It does not have a clean primitive for
authority** — the ability to say "X may do Y within constraints Z,
transferably, revocably, verifiably, without a trusted operator."

OAuth is a protocol hack around centralized IdPs. IAM is a database.
OPA is a rules engine. Zanzibar is Google's centralized ACL. Macaroons
(Google Research, 2014) came close but never had executable policies
or peer delegation.

The agent era breaks all of these. Agents delegate to agents to agents.
The authority graph explodes beyond anything RBAC can model. The
trust boundaries move from "user → server" to "user → local agent →
remote agent → third-party service." Every hop needs unforgeable,
scoped, auditable authorization.

This is a hole the shape of a new primitive. This document sketches
what that primitive could look like and how Leaves OS is positioned
to ship it.

## 1. What "Bitcoin-level novelty" means here

Bitcoin did not invent cryptography, P2P networking, or digital money.
It invented **digital scarcity without an issuer** — one specific
primitive that was thought impossible, required a distributed substrate,
and enabled previously-impossible behavior (self-custody, censorship
resistance, permissionless participation).

The analogous primitive for the agent era: **cryptographic authority
with executable semantics, transferable peer-to-peer, enforced at
the OS layer without a trusted operator.**

Bitcoin parallel:
- BTC = bearer asset with scarcity encoded in protocol
- AuthCap = bearer authority with policy encoded in protocol

Same shape of innovation: a cryptographic atom that replaces a
trusted third party.

## 2. The primitive: programmable capabilities

A capability is **a small WASM program + a cryptographic envelope**.
The program decides, at invocation time, whether a given action is
authorized, given dynamic context (time, caller identity, parameters,
prior invocations, revocation state). The envelope binds the program
to a principal's key.

### 2.1 Wire format (draft)

```
Capability {
  program: WASM,              // decide(context) -> Allow | Deny
  issuer:  Ed25519PublicKey,  // who signed this capability
  holder:  Ed25519PublicKey,  // who may invoke
  parent:  Option<blake3>,    // hash of parent capability (or None for root)
  nonce:   [u8; 16],          // replay protection
  issued:  u64,               // unix ms
  expires: u64,               // unix ms
  sig:     Ed25519Signature,  // issuer.sign(blake3(program || issuer || holder || parent || nonce || issued || expires))
}
```

Wire encoding: CBOR (tight, canonical, no JSON ambiguity).
Serialized size target: ≤2KB for typical capabilities including WASM.

### 2.2 Why WASM, not JSON scopes

JSON scopes can express "READ file" but not:
- "READ files matching `/home/user/finance/*` EXCEPT tagged confidential"
- "UNTIL 2026-05-01"
- "SO LONG AS today's cumulative READ count ≤ 50"
- "UNLESS caller has previously invoked DELETE in this session"

WASM encodes arbitrary predicates as a deterministic program. Runtime
is `wasmi` (or `wasmtime` in minimal mode): no imports, no syscalls,
gas-limited, deterministic. Target execution budget: <1ms per decision.

### 2.3 The WASM ABI (draft)

```
// Host provides:
//   host_get_context_json(ptr: *mut u8, cap: usize) -> usize
// Returns a JSON-serialized context into the buffer.

// Capability exports:
//   authorize() -> i32   // 0 = Deny, 1 = Allow, 2 = DelegateDown

// Context structure (JSON, canonical key order):
{
  "action": { "type": "research_query", "query": "...", ... },
  "caller": "<hex pubkey>",
  "timestamp": 1713196800123,
  "result": null | { "size_bytes": 1234, "hash": "<blake3>" },
  "invocation_count": 7,       // # of prior invocations of this capability
  "session_nonce": "<hex>"
}
```

Two-pass: policy runs once before the action (result=null) to permit,
then again after with result set to validate output properties
(size limits, no-PII patterns, etc).

### 2.4 Receipts

Every invocation produces a signed receipt:

```
Receipt {
  capability_hash: blake3,    // hash of the capability that authorized this
  action_hash:     blake3,    // hash of serialized action
  result_hash:     blake3,    // hash of result bytes
  executor:        Ed25519PublicKey,
  timestamp:       u64,
  runtime:         String,    // "leaves-0.5.0-abc123"
  attestation:     Option<TEEQuote>,  // Intel TDX / AMD SEV-SNP / Nitro — optional, Phase 2
  parent_receipt:  Option<blake3>,    // if this invocation was itself a sub-delegation
  sig:             Ed25519Signature,
}
```

Receipts chain. Given a final result, walk receipt.parent_receipt
backward to reconstruct the full delegation DAG.

### 2.5 Revocation

Issuers publish revocation hashes to a Leaves Revocation Registry
(LRR) implemented as an ICP canister. The canister stores an
append-only list of (capability_hash, revoked_at) entries. Clients
sync the registry every N minutes via certified queries.

For scale: the registry can be a Merkle tree of bloom filters
partitioned by time. A client keeps the last N filters locally,
fetches proofs for individual capability membership on demand.
Revocation check: O(1) bloom lookup + O(log N) Merkle proof
verification.

ICP is used specifically because:
- Certified queries give cryptographic proof of registry state
- No single node controls the registry
- Threshold ECDSA allows the registry itself to countersign
  high-value revocations
- Canisters are cheap to run continuously (vs deploying our own
  blockchain)

## 3. Identity

### 3.1 Root identity

Each device has a hardware-rooted Ed25519 keypair:
- TPM 2.0 on Linux (tpm2-tss, tpm2-tools)
- Secure Enclave on macOS/iOS
- Android Keystore with StrongBox on Android
- YubiHSM / Ledger as optional external root

Private key never leaves silicon. Public key is the principal's
long-lived identity.

### 3.2 Identity registry

Human-readable names map to pubkeys via an ICP canister:
- `alice.leaves` → `ed25519:0xabcd...`
- `bob.leaves/research` → `ed25519:0x1234...` (subpath = agent specialization)

Registration is first-come-first-serve with a small ICP cycles fee
to prevent squatting. Key rotation is supported: the canister stores
a rotation chain; clients verify the current key by walking the chain
from the last known good key.

### 3.3 Out-of-band exchange

For high-trust scenarios, pubkeys can be exchanged directly:
- QR code
- Safety numbers (per Signal)
- Shared token over a pre-existing secure channel

Identity registry is convenience, not required.

## 4. Transport

Peer-to-peer over QUIC + Noise_IK (mutual authentication with pre-known
remote public key). This gives:

- Forward secrecy
- Identity-pinned connections (no certificate authorities)
- Single round-trip handshake
- Multiplexed streams (for parallel delegations)

Wire protocol inside the Noise session:

```
LeavesWireMessage {
  kind: "invoke" | "receipt" | "error",
  capability: Option<Capability>,
  action: Option<Action>,
  receipt: Option<Receipt>,
  result: Option<bytes>,
  error: Option<ErrorCode>,
}
```

Peer discovery is out of scope for the core protocol. Options:
- Identity registry publishes endpoints (IP + port)
- mDNS for local network
- DHT (Kademlia-like) for at-scale discovery
- Manually configured peers for high-trust deployments

## 5. End-to-end example: two agents across the world

Setup:
- **Alice** in SF: Leaves OS, TPM-backed root key, registered `alice.leaves`.
- **Bob** in Berlin: Leaves OS server, TPM-backed root key, registered
  `bob.leaves/research`.
- Both running `leaves-0.5.0-abc123` (published hash).

### 5.1 Alice issues an intent

Alice types: *"Ask Bob's research agent for a Rust post-quantum
crypto library."*

Leaves parses → GoalSpec with a delegation step targeted at `bob.leaves/research`.

### 5.2 Alice's OS mints a capability

```
let program = wasm! {
    fn authorize(ctx) -> Decision {
        if ctx.action.type != "research_query" { return Deny }
        if ctx.timestamp > 1713196800 { return Deny }         // 1hr expiry
        if let Some(r) = ctx.result {
            if r.size_bytes > 4096 { return Deny }            // 4KB cap
        }
        if ctx.caller != BOB_PUBKEY { return Deny }
        Allow
    }
};

let cap = Capability::new(
    program,
    issuer = alice_pubkey,
    holder = bob_pubkey,
    parent = None,
    expires_in = 3600,
);
cap.sign(TPM.sign_fn(alice_tpm_handle));
```

TPM signs. Alice's private key never leaves silicon.

### 5.3 Transport: Alice → Bob

Alice's client:
1. Resolves `bob.leaves/research` via identity registry (or cache).
2. Establishes Noise_IK session to Bob's endpoint, pinning both keys.
3. Sends:
   ```
   InvokeMessage {
     capability: <bytes>,
     action: { type: "research_query", query: "Rust post-quantum crypto library" }
   }
   ```

### 5.4 Bob's runtime enforces

1. Verify `capability.sig` against `alice_pubkey` (fetched/cached).
2. Load capability WASM into wasmi, gas-limited.
3. Execute `authorize(ctx)` with action context, result=null → Allow.
4. Consult revocation registry (local cached bloom filter, fallback to
   ICP canister query). Not revoked.
5. Invoke Bob's research agent with the action. Agent produces 1.2KB
   of text.
6. Re-run `authorize(ctx)` with result populated → Allow (size ≤ 4KB).
7. Mint a receipt, sign with Bob's Ed25519 (TPM-backed).
8. Return `(receipt, result)` over the same Noise session.

### 5.5 Alice verifies

1. Verify `bob_pubkey` matches registry.
2. Verify `receipt.sig`.
3. Recompute `blake3(result)` and check against `receipt.result_hash`.
4. Verify `receipt.runtime` is in Alice's allowed-runtime list.
5. If attestation quote present, verify against Intel root cert
   (proves Bob's runtime binary matches the claimed hash).
6. Store receipt in audit log.
7. Render result with provenance UI: "Verified: bob.leaves/research,
   cap abc123, 2026-04-15 18:01:23."

### 5.6 Third-party audit

Given `(capability, receipt, result)`, anyone can:
- Verify Alice issued the capability (her signature, pubkey in registry)
- Verify Bob executed it (his signature, pubkey in registry)
- Verify result matches receipt hash (blake3)
- Rerun the WASM policy against the action to confirm authorization
- Check revocation status at the receipt's timestamp

No server, no account, no API key.

### 5.7 Composition (Bob sub-delegates to Carol)

Bob's agent needs translation. Carol in Tokyo specializes in it.

1. Bob mints a *child* capability:
   ```
   child_cap = Capability {
     program: wasm! { ... narrow scope: translate, en->* ... },
     issuer: bob_pubkey,
     holder: carol_pubkey,
     parent: Some(blake3(alice_cap)),
     ...
   };
   child_cap.sign(bob_tpm);
   ```
2. Bob invokes Carol with `child_cap`.
3. Carol's runtime verifies `child_cap.sig` against `bob_pubkey`, and
   walks `parent` back to confirm Alice's authorization covers this.
4. Carol executes, signs a receipt with `parent_receipt = None`
   (it's the leaf of the execution chain, but `capability.parent` is
   Bob's cap, which is child of Alice's).
5. Bob packages Carol's receipt into his receipt: `parent_receipt =
   Some(blake3(carol_receipt))`.
6. Alice gets Bob's receipt, walks the chain, verifies all signatures
   and all WASM policies all the way down.

The result is a cryptographic DAG of authority + execution that
no party could forge without the others' keys.

## 6. What falls out of this primitive for free

### 6.1 Sovereign data capsules

Encode `program` as: "given ciphertext C and key K, decrypt C and
expose it to the caller's agent ONLY IF policy satisfied."

Data travels with its rules. Recipient can query it but cannot extract
the raw bytes outside policy. Patient records, legal documents, financial
data — all become bearer objects with self-enforcing access.

### 6.2 Private data oracles

Your agent indexes your data. Third parties pay for queries via
capabilities: "is my income > $50K?" Your agent answers with a ZK
proof. They never see the raw data.

Monetize personal data without releasing it.

### 6.3 Agent-to-agent markets

Agents advertise capability templates: "I'll accept this class of
capability, produce results matching this spec, for X ICP cycles per
call." Discovery via identity registry tags. Payments via capability
co-signing + escrow canister.

### 6.4 Unforgeable agent reputation

Bob's pubkey accumulates N receipts from distinct issuers. Each
receipt is evidence Bob executed as promised. Reputation is a
function of the signed history. Can't be forged (no valid signatures),
can't be transferred (bound to Bob's key), but can be demonstrably
used (Bob proves possession of his key).

### 6.5 Supply-chain provenance for AI output

Every AI-generated document carries its receipt DAG as a metadata
header. "Was this contract draft prepared by an authorized agent
using approved models on data the owner consented to share?" —
walk the DAG, verify every edge.

Regulatory compliance (HIPAA, GDPR, SOX) becomes structural rather
than paperwork.

### 6.6 Non-repudiation and liability

If Bob's agent produces malicious output, Bob's signature is on it.
Alice has proof. Insurance companies, courts, auditors all have
mathematical evidence. No "he said, she said."

## 7. Phased implementation plan

### Phase 1 — Local enforcement (2 weeks)
- Runtime enforcer on top of Leaves agents
- Capability schema (CBOR serialization, signing, verification)
- WASM policy VM (wasmi) integrated with agent dispatch
- Red-team test suite: 30+ adversarial plans validating rejection
- Audit log records capability_hash and policy result for every action

Deliverable: "Leaves OS enforces action contracts at the syscall
boundary. Agents cannot execute actions outside their plan."

### Phase 2 — Cross-device delegation (2 weeks)
- Ed25519 keypair generation via TPM 2.0
- Identity registry (initial: shared JSON file or simple canister)
- Noise_IK transport layer
- Capability minting + signing + transmission
- Receipt chain reconstruction
- Time-machine UI shows the delegation DAG

Deliverable: "Two Leaves devices can delegate tasks to each other
with full cryptographic provenance."

### Phase 3 — Production hardening (1-2 months)
- TEE attestation (Intel TDX or AMD SEV-SNP)
- Revocation registry on ICP
- Identity registry on ICP with key rotation
- Threshold ECDSA for canister-countersigned high-value caps
- Formal security review of capability schema
- Reference implementation in Rust (not just the Python prototype)

### Phase 4 — Ecosystem (6+ months)
- BIP-01-equivalent: Leaves Authority Protocol spec, versioned
- Reference WASM policy library (common patterns: time-bounded,
  rate-limited, data-class-restricted)
- Integration with third-party agent frameworks (Ollama, LangChain,
  MCP bridges)
- Capability marketplace proof-of-concept
- Private data oracle demo (e.g., personal LLM answers income-range
  queries with ZK proofs)

## 8. Honest risk assessment

### 8.1 Adoption risk (the big one)

Protocols die without a compelling reason for the *first user* to
adopt them without an ecosystem. Bitcoin survived its first year
because 10 cypherpunks genuinely needed censorship-resistant money.
Macaroons didn't because nobody specifically needed them over OAuth.

The capability protocol's first user is a Leaves OS user invoking
a local agent. The protocol must be useful *before* network effects —
Phase 1 must stand alone as product value ("AI OS with runtime-
enforced action contracts"). Phase 2 onward trades on distribution
earned in Phase 1.

### 8.2 Technical risks

| Risk | Mitigation |
|------|-----------|
| WASM policy escape | wasmi sandbox, gas limits, no syscall imports |
| TEE attestation bugs | Optional in Phase 1-2; reproducible builds + hash pinning as fallback |
| Key compromise | TPM-backed keys; key rotation via identity registry |
| Replay attacks | Nonce + session binding in receipts |
| Capability schema churn | Versioned schema; forward-compat by ignoring unknown fields |
| Revocation liveness | Bloom filter + Merkle tree; clients cache aggressively, trade freshness for availability |
| ICP dependency | Replaceable; canister functionality can run on Substrate, Cosmos, or a federated non-blockchain log |

### 8.3 Positioning risk

If we pitch "agent-to-agent protocol" too early, people think
blockchain-grift. The sales order must be:
1. Local enforcer works (concrete safety win)
2. Hardware identity works (concrete provenance win)
3. Delegation works (novel capability)
4. Receipt chains work (regulatory/compliance win)
5. Then, quietly, the protocol emerges

Lead with utility. Let the infrastructure story be the ask-me-more.

### 8.4 Competitive risk

Players likely to ship something adjacent in 12-24 months:
- Anthropic (MCP already in this direction, missing peer delegation
  and hardware root)
- Google (could extend Zanzibar / Passkeys into the agent domain)
- Apple (Private Cloud Compute is the closest production system; if
  they open APIs, we're commoditized)
- A well-funded startup (Figma-scale team, 12-month head start wasted)

Our defensible moat: being first to specify and ship the open version.
Standards adoption is a winner-take-most game; second place is
irrelevant. This is also why the spec matters more than the
implementation — if the spec is clean and permissively licensed,
it can spread even if our reference implementation loses.

## 9. Outcome distribution (honest numbers)

| Scenario | Users (5yr) | Comparable | Probability |
|---|---|---|---|
| New standard | 10⁸+ | TLS, DNS, Ed25519 | ~5% |
| Strong niche | 10⁶–10⁷ | WireGuard, Noise | ~20% |
| Academically cited, unused | 10³–10⁴ | Macaroons, SPKI | ~75% |

Expected value is very high because the best case is enormous. But
this is an infrastructure bet: high variance, long time horizon,
winner-take-most.

## 10. Open design questions

1. **Policy composition.** Can multiple capabilities be AND-ed into
   an effective policy? (E.g., user's daily quota AND the specific
   task scope.) Current design: no, each invocation presents one
   capability. Consider: stacked capabilities where runtime evaluates
   intersection.

2. **Delegation with scope reduction.** How does Bob prove his child
   capability's scope is strictly narrower than Alice's parent? Options:
   (a) runtime just checks both and requires both to allow,
   (b) formal subset verifier (hard), (c) template-based reductions
   (simple but limiting). Current design favors (a).

3. **Confidentiality.** Capability programs may leak information
   (e.g., the policy reveals what data exists). Do we need
   encrypted capabilities readable only by the holder? Probably
   yes for Phase 3+.

4. **Quantum resistance.** Ed25519 is not post-quantum secure.
   When do we migrate to Dilithium or Falcon? Probably along with
   the broader industry migration. Schema should be algorithm-agile.

5. **Audit log anchoring.** Should receipts be periodically
   Merkle-anchored to a public ledger for non-repudiation beyond
   individual signatures? If so, what ledger, and how to prevent
   the ledger from becoming a trust bottleneck?

6. **Key loss / recovery.** If Alice loses her TPM, she loses her
   identity. Social recovery? Threshold signing across multiple
   devices? Still an open UX problem; borrow from the FIDO2 /
   Passkeys approach.

## 11. Glossary

- **Capability**: signed authority atom; WASM program + envelope
- **Receipt**: signed record of an invocation
- **Principal**: entity identified by a public key (human, device, agent)
- **Issuer**: principal who minted a capability
- **Holder**: principal authorized to invoke a capability
- **Delegation**: issuing a child capability derived from a parent
- **LRR**: Leaves Revocation Registry (ICP canister)
- **Runtime**: the software that enforces capabilities on a device
- **Attestation**: cryptographic proof that a runtime binary matches
  a claimed hash, produced by hardware (TEE)

## 12. References

- Macaroons: Google Research, 2014. "Macaroons: Cookies with Contextual
  Caveats for Decentralized Authorization in the Cloud"
- Noise Protocol: https://noiseprotocol.org
- ICP threshold ECDSA: https://internetcomputer.org/docs/current/developer-docs/integrations/t-ecdsa/
- Intel TDX: https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/
- Passkeys / FIDO2: https://fidoalliance.org/passkeys/
- wasmi: https://github.com/paritytech/wasmi
- BLAKE3: https://github.com/BLAKE3-team/BLAKE3
- CBOR: RFC 8949

---

This document is a living sketch. Revise as the protocol matures.
