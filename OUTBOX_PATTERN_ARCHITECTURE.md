# Outbox Pattern với Redis Stream - Kiến trúc và Cách hoạt động

## 📋 Tổng quan

Outbox Pattern là một pattern để đảm bảo **reliable event publishing** trong microservices architecture. Pattern này đảm bảo rằng:
- Events được lưu trong database **cùng transaction** với domain changes
- Events được publish một cách **đáng tin cậy** và **không bị mất**
- Có khả năng **retry** khi publish thất bại

---

## 🏗️ Kiến trúc

```
┌─────────────────┐
│  Service Layer  │
│  (JobService)   │
└────────┬────────┘
         │
         │ 1. Save Entity + OutboxEvent (same transaction)
         ▼
┌──────────────────────────┐
│   PostgreSQL Database    │
│  ┌────────────────────┐  │
│  │  jobs              │  │
│  │  outbox_events     │◄─┼── Status: PENDING
│  └────────────────────┘  │
└──────────────────────────┘
         │
         │ 2. Scheduled Processor (every 5 seconds)
         ▼
┌──────────────────────────┐
│  OutboxEventProcessor    │
│  - Reads PENDING events  │
│  - Publishes to Redis    │
│  - Updates status        │
└───────────┬──────────────┘
            │
            │ 3. Publish to Stream
            ▼
┌──────────────────────────┐
│      Redis Stream         │
│  Stream: outbox:events    │
│  ┌──────────────────────┐ │
│  │ Message 1: CREATED   │ │
│  │ Message 2: UPDATED   │ │
│  │ Message 3: DELETED  │ │
│  └──────────────────────┘ │
└───────────┬──────────────┘
            │
            │ 4. Consumer reads from stream
            ▼
┌──────────────────────────┐
│   Python Service         │
│  - Consumes messages     │
│  - Syncs to Milvus       │
│  - Acknowledges messages │
└──────────────────────────┘
```

---

## 🔄 Luồng hoạt động chi tiết

### **Bước 1: Tạo Event trong Transaction**

Khi service tạo/cập nhật/xóa entity, nó lưu cả entity VÀ outbox event trong **cùng một transaction**:

```java
@Transactional
public JobResponse createJob(Account account, CreateJobRequest request) {
    // 1. Save domain entity
    Job job = jobRepository.save(new Job(...));
    
    // 2. Convert to DTO
    JobResponse jobResponse = jobMapper.toResponse(job);
    
    // 3. Save outbox event (same transaction!)
    String payload = objectMapper.writeValueAsString(jobResponse);
    outboxEventService.saveOutboxEvent(
        "JOB",           // aggregateType
        job.getId(),     // aggregateId
        "CREATED",       // eventType
        payload          // JSON string
    );
    
    // ✅ Nếu transaction commit → cả job và outbox event đều được lưu
    // ❌ Nếu transaction rollback → cả job và outbox event đều bị hủy
    return jobResponse;
}
```

**Database State sau Bước 1:**
```
outbox_events table:
┌────┬──────────────┬─────────────┬──────────┬──────────┬─────────┐
│ id │aggregateType │aggregateId  │eventType │  status  │attempts │
├────┼──────────────┼─────────────┼──────────┼──────────┼─────────┤
│123 │ "JOB"        │     456     │ "CREATED"│ "PENDING"│   0     │
└────┴──────────────┴─────────────┴──────────┴──────────┴─────────┘
```

### **Bước 2: Background Processor**

`OutboxEventProcessor` chạy **mỗi 5 giây** để:
1. Tìm tất cả events có status = `PENDING`
2. Publish từng event vào Redis Stream
3. Update status thành `SENT` (nếu thành công) hoặc `FAILED` (nếu thất bại)

```java
@Scheduled(fixedDelay = 5000, initialDelay = 10000)
@Transactional
public void processPendingEvents() {
    // 1. Query PENDING events (with pessimistic lock)
    List<OutboxEvent> pendingEvents = 
        outboxEventRepository.findPendingEvents(OutboxStatus.PENDING);
    
    // 2. Process each event
    for (OutboxEvent event : pendingEvents) {
        boolean success = redisStreamPublisher.publishToStream(event);
        
        if (success) {
            event.setStatus(OutboxStatus.SENT);  // ✅ Success
        } else {
            event.setAttempts(event.getAttempts() + 1);
            if (event.getAttempts() >= 3) {
                event.setStatus(OutboxStatus.DLQ);  // ❌ Dead Letter Queue
            } else {
                event.setStatus(OutboxStatus.FAILED);  // ⚠️ Will retry
            }
        }
        outboxEventRepository.save(event);
    }
}
```

**Pessimistic Lock:**
- Sử dụng `@Lock(LockModeType.PESSIMISTIC_WRITE)` để tránh nhiều instance xử lý cùng event
- Đảm bảo **exactly-once processing**

### **Bước 3: Publish to Redis Stream**

`RedisStreamPublisherImpl` tạo message với **8 fields** và publish vào Redis Stream:

```java
public boolean publishToStream(OutboxEvent event) {
    // 1. Build message fields
    Map<String, String> fields = new HashMap<>();
    fields.put("id", String.valueOf(event.getId()));
    fields.put("aggregateType", event.getAggregateType());
    fields.put("aggregateId", String.valueOf(event.getAggregateId()));
    fields.put("eventType", event.getEventType());
    fields.put("payload", event.getPayload());  // Full entity data as JSON
    fields.put("occurredAt", event.getOccurredAt().toString());
    fields.put("traceId", event.getTraceId().toString());
    fields.put("attempts", String.valueOf(event.getAttempts()));
    
    // 2. Create stream record
    var record = StreamRecords.newRecord()
            .ofStrings(fields)
            .withStreamKey("outbox:events");
    
    // 3. Publish to Redis Stream
    RecordId recordId = redisTemplate.opsForStream().add(record);
    
    return recordId != null;  // Success if recordId is not null
}
```

**Redis Stream State sau Bước 3:**
```
Stream: outbox:events
Message ID: 1705291200000-0
Fields:
  id: "123"
  aggregateType: "JOB"
  aggregateId: "456"
  eventType: "CREATED"
  payload: "{\"id\":456,\"title\":\"Software Engineer\",...}"
  occurredAt: "2024-01-15T10:30:00+07:00"
  traceId: "550e8400-e29b-41d4-a716-446655440000"
  attempts: "0"
```

### **Bước 4: Python Service Consumes**

Python service đọc messages từ Redis Stream và sync vào Milvus:

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Create consumer group
try:
    r.xgroup_create("outbox:events", "milvus-sync", id="0", mkstream=True)
except:
    pass  # Group already exists

# Consume messages
while True:
    messages = r.xreadgroup(
        "milvus-sync",
        "worker-1",
        {"outbox:events": ">"},
        count=10,
        block=1000
    )
    
    for stream, msgs in messages:
        for msg_id, fields in msgs:
            # Parse event
            event_type = fields["eventType"]
            aggregate_type = fields["aggregateType"]
            payload = json.loads(fields["payload"])
            
            # Sync to Milvus
            if aggregate_type == "JOB":
                if event_type in ["CREATED", "UPDATED"]:
                    sync_job_to_milvus(payload)
                elif event_type == "DELETED":
                    delete_job_from_milvus(fields["aggregateId"])
            
            # Acknowledge message
            r.xack("outbox:events", "milvus-sync", msg_id)
```

---

## 🧩 Các Components

### 1. **OutboxEventService** (Interface)
- Interface để save outbox events
- 2 methods:
  - `saveOutboxEvent(aggregateType, aggregateId, eventType, payload)` - Auto generate traceId
  - `saveOutboxEvent(aggregateType, aggregateId, eventType, payload, traceId)` - Custom traceId

### 2. **OutboxEventServiceImpl**
- Implementation của `OutboxEventService`
- Lưu event vào database với status = `PENDING`
- Sử dụng `@Transactional` để đảm bảo atomic

### 3. **OutboxEventProcessor**
- **Scheduled component** chạy mỗi 5 giây
- Đọc PENDING events từ database
- Publish events vào Redis Stream
- Update status: `PENDING` → `SENT` / `FAILED` → `DLQ`

### 4. **RedisStreamPublisher**
- Interface để publish events vào Redis Stream
- Method: `publishToStream(OutboxEvent event)`

### 5. **RedisStreamPublisherImpl**
- Implementation của `RedisStreamPublisher`
- Tạo stream record với 8 fields
- Publish vào Redis Stream (`outbox:events`)

### 6. **OutboxEventRepository**
- JPA Repository với query methods
- `findPendingEvents(OutboxStatus.PENDING)` - Query với pessimistic lock

---

## 📊 Event States & Lifecycle

```
┌─────────┐
│ PENDING│ ← Created when event is saved
└───┬─────┘
    │
    │ Processed by OutboxEventProcessor
    │
    ├─────────────────┬──────────────────┐
    │                 │                  │
    ▼                 ▼                  ▼
┌───────┐      ┌─────────┐      ┌─────────┐
│  SENT │      │ FAILED  │      │   DLQ   │
│       │      │         │      │         │
│✅Done │      │⚠️Retry  │      │❌Error  │
└───────┘      └────┬────┘      └─────────┘
                    │
                    │ After 3 failed attempts
                    └──────────► DLQ
```

**Status Definitions:**
- **PENDING**: Event mới tạo, chưa được publish
- **SENT**: Event đã được publish thành công vào Redis Stream
- **FAILED**: Publish thất bại, sẽ retry (attempts < 3)
- **DLQ**: Dead Letter Queue - Publish thất bại sau 3 lần thử

---

## 🔒 Transactional Guarantees

### Atomicity
- Entity save và OutboxEvent save trong **cùng transaction**
- Nếu một trong hai thất bại → cả hai đều rollback

### Consistency
- Outbox event chỉ được tạo khi entity được lưu thành công
- Không có orphan events

### Isolation
- Sử dụng **Pessimistic Lock** khi query PENDING events
- Đảm bảo không có race condition giữa các instances

### Durability
- Events được lưu trong PostgreSQL (durable storage)
- Không bị mất ngay cả khi application crash

---

## 🚀 Retry Mechanism

### Automatic Retry
- Failed events được đánh dấu `FAILED`
- Processor sẽ retry events có status `FAILED` trong lần chạy tiếp theo
- Maximum 3 attempts

### Dead Letter Queue (DLQ)
- Sau 3 lần thử thất bại → chuyển sang `DLQ`
- DLQ events cần được xử lý manually hoặc alert

---

## 📝 Example: Complete Flow

### Scenario: Create a new Job

**1. User creates job via API:**
```
POST /api/jobs
{
  "title": "Software Engineer",
  "company": "Acme Corp",
  ...
}
```

**2. JobServiceImpl.createJob():**
```java
@Transactional
public JobResponse createJob(...) {
    // Save job
    Job job = jobRepository.save(new Job(...));
    
    // Save outbox event
    JobResponse response = mapper.toResponse(job);
    String payload = objectMapper.writeValueAsString(response);
    outboxEventService.saveOutboxEvent("JOB", job.getId(), "CREATED", payload);
    
    return response;  // Transaction commits here
}
```

**3. Database state:**
```sql
-- jobs table
INSERT INTO jobs (id, title, ...) VALUES (456, 'Software Engineer', ...);

-- outbox_events table
INSERT INTO outbox_events 
  (id, aggregate_type, aggregate_id, event_type, payload, status, attempts)
VALUES 
  (123, 'JOB', 456, 'CREATED', '{"id":456,...}', 'PENDING', 0);
```

**4. OutboxEventProcessor (after 5 seconds):**
- Queries: `SELECT * FROM outbox_events WHERE status = 'PENDING'`
- Finds event id=123
- Publishes to Redis Stream
- Updates status to 'SENT'

**5. Redis Stream:**
```
Stream: outbox:events
Message: 1705291200000-0
  id: "123"
  aggregateType: "JOB"
  aggregateId: "456"
  eventType: "CREATED"
  payload: "{\"id\":456,\"title\":\"Software Engineer\",...}"
  ...
```

**6. Python Service:**
- Consumes message from stream
- Parses payload JSON
- Syncs job data to Milvus vector database
- Acknowledges message

---

## ⚙️ Configuration

### Application Properties
```properties
# Redis Stream key (default: outbox:events)
outbox.redis.stream.key=outbox:events

# Redis connection
spring.data.redis.host=localhost
spring.data.redis.port=6379
```

### Scheduler Configuration
- **Frequency**: Every 5 seconds (`fixedDelay = 5000`)
- **Initial Delay**: 10 seconds (`initialDelay = 10000`)
- **Retry Attempts**: Maximum 3 attempts

---

## 🔍 Monitoring & Debugging

### Check Pending Events
```sql
SELECT * FROM outbox_events 
WHERE status = 'PENDING' 
ORDER BY occurred_at ASC;
```

### Check Failed Events
```sql
SELECT * FROM outbox_events 
WHERE status IN ('FAILED', 'DLQ') 
ORDER BY occurred_at DESC;
```

### Check Redis Stream
```bash
# Stream info
redis-cli XINFO STREAM outbox:events

# Read messages
redis-cli XREAD COUNT 10 STREAMS outbox:events 0

# Consumer groups
redis-cli XINFO GROUPS outbox:events
```

---

## ✅ Best Practices

1. **Always use @Transactional** when saving outbox events
2. **Handle exceptions gracefully** - don't let outbox failures break main flow
3. **Use meaningful event types** - CREATED, UPDATED, DELETED, PUBLISHED
4. **Include full entity data** in payload for downstream services
5. **Monitor DLQ** - Alert when events move to DLQ
6. **Test event publishing** - Verify events appear in Redis Stream
7. **Use consumer groups** in Python service for reliable consumption

---

## 🎯 Benefits

✅ **Reliability**: Events guaranteed to be saved before transaction commits
✅ **Atomicity**: Entity and event save in same transaction
✅ **Durability**: Events stored in PostgreSQL (no data loss)
✅ **Retry Logic**: Automatic retry for failed publishes
✅ **Scalability**: Can handle high throughput with background processing
✅ **Decoupling**: Producer and consumer are decoupled via Redis Stream

---

## 🛠️ Troubleshooting

### Events stuck in PENDING
- Check if `OutboxEventProcessor` is running
- Check scheduler is enabled: `@EnableScheduling`
- Check logs for errors

### Events in DLQ
- Check Redis connection
- Check Redis Stream is accessible
- Manual retry: Update status from DLQ to PENDING

### Missing events in Redis Stream
- Check `RedisStreamPublisher` logs
- Verify Redis connection
- Check stream key configuration

---

## 📚 Summary

Outbox Pattern với Redis Stream đảm bảo:
1. ✅ **Reliable event publishing** - Events được lưu trong DB trước
2. ✅ **Transactional consistency** - Entity và event trong cùng transaction
3. ✅ **Background processing** - Không block main flow
4. ✅ **Automatic retry** - Retry failed publishes
5. ✅ **Scalable consumption** - Python service consume từ Redis Stream
6. ✅ **Milvus sync** - Python service sync data vào Milvus vector database

