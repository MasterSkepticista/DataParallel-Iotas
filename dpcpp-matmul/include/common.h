#pragma once
/*
	TODO:
	[x]: Use namespaces for timing/CL functions separately
*/
#include <chrono>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include "settings.h"

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name, mode) pfr::InstrumentationTimer timer##__LINE__(name, mode) // Same timernames in one fn: append LINE_NUMBER!!!
#define PROFILE_FUNCTION(mode) PROFILE_SCOPE(__FUNCSIG__, mode) // __FUNCTION__ only fn name, __FUNCSIG__ shows overloads
#endif

#define GIGAFLOPS(x) (2*(pow(SIZE, 3))*1e-9 / x)

namespace pfr {
	struct ProfileResult {
		// Each profile entity will have Name, Starttime, Endtime
		std::string Name;
		long long Start, End;
	};

	struct InstrumentationSession {
		// TODO: Why?
		std::string Name;
	};

	class Instrumentor {
		// This class will format the output and dump it to a JSON file
	private:
		InstrumentationSession* m_CurrentSession;
		std::ofstream m_OutputStream;
		int m_ProfileCount;

	public:
		// This is a good practice of initializing values in a constructor
		// You can place empty braces too
		Instrumentor()
			: m_CurrentSession(nullptr), m_ProfileCount(0)
		{

		}

		void BeginSession(const std::string& name, const std::string& filepath = "results.json") {
			m_OutputStream.open(filepath);
			WriteHeader();
			m_CurrentSession = new InstrumentationSession{ name };
		}

		void EndSession() {
			WriteFooter();
			m_OutputStream.close();
			delete m_CurrentSession;
			m_CurrentSession = nullptr;
			m_ProfileCount = 0;
		}

		void WriteProfile(const ProfileResult& result) {
			if (m_ProfileCount++ > 0)
				m_OutputStream << ", ";

			std::string name = result.Name;
			std::replace(name.begin(), name.end(), '"', '\'');

			/*
			This is essentially printing a dictionary item in the format:
			{
				"cat": "function",
				"dur": 10,
				"name": "multiplyGPU",
				"ph": "X",
				"pid": 0,
				"tid": 0,
				"ts": 5
			}
			*/
			m_OutputStream << "{";
			m_OutputStream << "\"cat\": \"function\", ";
			m_OutputStream << "\"dur\": " << (result.End - result.Start) << ",";
			m_OutputStream << "\"name\": \"" << name << "\",";
			m_OutputStream << "\"ph\": \"X\",";
			m_OutputStream << "\"pid\": 0,";
			m_OutputStream << "\"tid\": 0,";
			m_OutputStream << "\"ts\": " << result.Start;
			m_OutputStream << "}";

			m_OutputStream.flush();
		}

		void WriteHeader() {
			m_OutputStream << "{\"otherData\": {}, \"traceEvents\": [";
			m_OutputStream.flush();
		}

		void WriteFooter() {
			m_OutputStream << "]}";
			m_OutputStream.flush();
		}

		static Instrumentor& Get() {
			static Instrumentor* instance = new Instrumentor();
			return *instance;
		}
	};

	class InstrumentationTimer {
		// Scope timing class that follows RAII: Resource Acquisition Is Initialization
	public:
		InstrumentationTimer(const char* name, std::string mode = "time")
			: m_Name(name),  m_Mode(mode), m_Stopped(false)
		{
			m_StartTimePoint = std::chrono::high_resolution_clock::now();
		}

		~InstrumentationTimer() {
			if (!m_Stopped)
				Stop();
		}

		void Stop() {
			auto endTimePoint = std::chrono::high_resolution_clock::now();

			// Cast the gonzo number to microseconds
			auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimePoint).time_since_epoch().count();
			auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();

			auto t_Seconds = (end - start) * 1e-6;
			if (m_Mode == "time")
				std::cout << m_Name << ": " << t_Seconds << "s\n";
			else if (m_Mode == "gflops")
				std::cout << m_Name << ": " << GIGAFLOPS(t_Seconds) << " GFLOPS (" << t_Seconds << "s)\n";
			Instrumentor::Get().WriteProfile({ m_Name, start, end });
		}

	private:
		const char* m_Name;
		std::string m_Mode;
		bool m_Stopped;
		std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
	};
}