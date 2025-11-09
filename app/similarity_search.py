from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer

def diagnose_database():
    """Check database state before running searches"""
    try:
        vec = VectorStore()
        
        print("=== DATABASE DIAGNOSIS ===")
        
        # Check table structures
        print("\n1. Table Structures:")
        vec.cur.execute("""
            SELECT table_name, column_name, data_type 
            FROM information_schema.tables 
            JOIN information_schema.columns USING (table_schema, table_name)
            WHERE table_schema = 'public' 
            AND table_name IN ('embeddings', 'test_embeddings', 'documents')
            ORDER BY table_name, ordinal_position
        """)
        columns = vec.cur.fetchall()
        
        if not columns:
            print("   No tables found with expected names")
        else:
            for table, column, dtype in columns:
                print(f"   {table}.{column} ({dtype})")
        
        # Check data samples
        print("\n2. Data Samples:")
        tables_to_check = ['embeddings', 'test_embeddings', 'documents']
        total_count = 0
        found_tables = []
        
        for table in tables_to_check:
            try:
                vec.cur.execute(f"SELECT COUNT(*) as count FROM {table}")
                count_result = vec.cur.fetchone()
                if count_result:
                    count = count_result[0]
                    print(f"   {table}: {count} rows")
                    total_count += count
                    found_tables.append(table)
                    
                    if count > 0:
                        vec.cur.execute(f"SELECT id, content FROM {table} LIMIT 2")
                        samples = vec.cur.fetchall()
                        print(f"   Sample data: {samples}")
                else:
                    print(f"   {table}: Could not count rows")
            except Exception as e:
                print(f"   Error checking table {table}: {e}")
        
        vec.close()
        return total_count > 0, found_tables
        
    except Exception as e:
        print(f"Error during database diagnosis: {e}")
        return False, []

def main():
    # First, diagnose the database
    has_data, available_tables = diagnose_database()
    
    if not has_data:
        print("\nNo data found in tables. Please add some embeddings first.")
        print("Run the data ingestion script to populate the database.")
        return
    
    print(f"Available tables with data: {available_tables}")
    
    # Initialize VectorStore for searching
    try:
        vec = VectorStore()
    except Exception as e:
        print(f"Error initializing VectorStore: {e}")
        return
    
    print("\n" + "="*50)
    print("STARTING SIMILARITY SEARCH TESTS")
    print("="*50)

    # --------------------------------------------------------------
    # Test 1: Basic shipping question
    # --------------------------------------------------------------
    print("\nTEST 1: Shipping Question")
    print("-" * 30)
    
    relevant_question = "What are your shipping options?"
    print(f"Question: {relevant_question}")
    
    try:
        results = vec.search(relevant_question, limit=3)
        print(f"Found {len(results)} results")
        
        if len(results) > 0:
            print("Sample results:")
            for i, result in enumerate(results.head(2).itertuples()):
                print(f"  {i+1}. {getattr(result, 'content', 'No content')[:100]}...")
            
            response = Synthesizer.generate_response(question=relevant_question, context=results)
            
            print(f"\nAnswer: {response.answer}")
            print("\nThought process:")
            for thought in response.thought_process:
                print(f"- {thought}")
            print(f"\nContext sufficient: {response.enough_context}")
        else:
            print("No results found for this query.")
        
    except Exception as e:
        print(f"Error in Test 1: {e}")
        import traceback
        traceback.print_exc()

    # --------------------------------------------------------------
    # Test 2: Irrelevant question
    # --------------------------------------------------------------
    print("\nTEST 2: Irrelevant Question")
    print("-" * 30)
    
    irrelevant_question = "What is the weather in Tokyo?"
    print(f"Question: {irrelevant_question}")
    
    try:
        results = vec.search(irrelevant_question, limit=3)
        print(f"Found {len(results)} results")
        
        if len(results) > 0:
            print("Sample results:")
            for i, result in enumerate(results.head(2).itertuples()):
                print(f"  {i+1}. {getattr(result, 'content', 'No content')[:100]}...")
            
            response = Synthesizer.generate_response(question=irrelevant_question, context=results)
            
            print(f"\nAnswer: {response.answer}")
            print("\nThought process:")
            for thought in response.thought_process:
                print(f"- {thought}")
            print(f"\nContext sufficient: {response.enough_context}")
        else:
            print("No results found for this query.")
        
    except Exception as e:
        print(f"Error in Test 2: {e}")
        import traceback
        traceback.print_exc()

    # --------------------------------------------------------------
    # Test 3: Metadata filtering
    # --------------------------------------------------------------
    print("\nTEST 3: Metadata Filtering")
    print("-" * 30)
    
    metadata_filter = {"category": "Shipping"}
    print(f"Metadata filter: {metadata_filter}")
    
    try:
        results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)
        print(f"Found {len(results)} results")
        
        if len(results) > 0:
            print("Sample results:")
            for i, result in enumerate(results.head(2).itertuples()):
                print(f"  {i+1}. {getattr(result, 'content', 'No content')[:100]}...")
                if hasattr(result, 'category'):
                    print(f"     Category: {getattr(result, 'category', 'N/A')}")
            
            response = Synthesizer.generate_response(question=relevant_question, context=results)
            
            print(f"\nAnswer: {response.answer}")
            print("\nThought process:")
            for thought in response.thought_process:
                print(f"- {thought}")
            print(f"\nContext sufficient: {response.enough_context}")
        else:
            print("No results found with this metadata filter.")
        
    except Exception as e:
        print(f"Error in Test 3: {e}")
        import traceback
        traceback.print_exc()

    # --------------------------------------------------------------
    # Test 4: Time-based filtering
    # --------------------------------------------------------------
    print("\nTEST 4: Time-based Filtering")
    print("-" * 30)
    
    try:
        # Recent dates — Should return results
        time_range = (datetime(2024, 1, 1), datetime(2024, 12, 31))
        results = vec.search(relevant_question, limit=3, time_range=time_range)
        print(f"2024 dates - Found {len(results)} results")
        
        # Old dates — Might not return results
        time_range = (datetime(2020, 1, 1), datetime(2020, 12, 31))
        results = vec.search(relevant_question, limit=3, time_range=time_range)
        print(f"2020 dates - Found {len(results)} results")
        
    except Exception as e:
        print(f"Error in Test 4: {e}")
        import traceback
        traceback.print_exc()

    # Close connection
    try:
        vec.close()
    except:
        pass
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()